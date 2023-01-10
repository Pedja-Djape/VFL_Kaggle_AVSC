from typing import Callable, Union, Optional, List, Tuple, Dict

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)

from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

import numpy as np
import model
from sklearn.metrics import f1_score

from torch import optim
from torch.nn import BCELoss
import torch

from pickle import dump,loads
accs = []

torch.manual_seed(0)
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

class SplitVFL(Strategy):
    def __init__(self, num_clients, batch_size, dim_input, train_labels, test_labels, lr=1e-6):
        super(SplitVFL, self).__init__()

        self.batch_size = batch_size
        self.num_clients = num_clients
        self.dim_input = dim_input
        self.lr = lr
        self.criterion = BCELoss()
        self.train_labels = torch.tensor(train_labels)
        self.test_labels  = torch.tensor(test_labels)

        self.train_lwr_idx = 0
        self.test_lwr_idx = 0

        self.train_acc = []
        self.test_acc = []

        self.train_correct = 0
        self.test_correct = 0

        self.test_f1 = []
        self.test_f1_accum = 0

        self.model = None
        self.client_tensors = None
        self.clients_map = None
        self.optim = None

        
        
    def __repr__(self) -> str:
        return "VerticalFed"
    
    def update_label_index(self,test=False):
        def update(lwr,len_ls):
            l,u = None,None
            if lwr < len_ls:
                if len_ls - lwr < self.batch_size:
                    u = len_ls
                    l = lwr
                else:
                    u = lwr + self.batch_size
                    l = lwr
            else:
                l = 0
                u = l + self.batch_size
                if test:
                    self.test_acc += [self.test_correct / len(self.test_labels)]
                    self.test_correct = 0
                    self.test_f1 += [self.test_f1_accum / len(self.test_labels)]
                    self.test_f1_accum
                else:
                    self.train_acc += [self.train_correct / len(self.train_labels)]
                    self.train_correct = 0
            return l,u
            
        
        idx_u = None
        if not test:
            self.train_lwr_idx, idx_u = update(lwr=self.train_lwr_idx, len_ls=len(self.train_labels))
        else:
            self.test_lwr_idx, idx_u = update( lwr=self.test_lwr_idx,  len_ls=len(self.test_labels))
        # if self.train_lwr_idx < len(self.train_labels):
        #     # last batch has less than batch_size samples
        #     if len(self.train_labels) - self.train_lwr_idx < self.batch_size:
        #         idx_u = len(self.train_labels)
        #     else:
        #         # update upper bound normally
        #         idx_u = self.train_lwr_idx + self.batch_size
        # else:
        #     # just passed final batch
        #     self.train_lwr_idx = 0
        #     idx_u = self.train_lwr_idx + self.batch_size
        
        return idx_u
    
    def initialize_parameters(
        self, client_manager: ClientManager
        ) -> Optional[Parameters]:
        # Initialize global model parameters
        global_model = model.Net(n_inputs=self.dim_input, output_dim=1)
        ndarrays = get_parameters(net=global_model)

        self.model = global_model
        self.optim = optim.Adam(self.model.parameters(),lr=self.lr)
        self.criterion = BCELoss()
        return ndarrays_to_parameters(ndarrays)
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
        ) -> List[Tuple[ClientProxy,FitIns]]:
        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients,
        )
        config = { 'server_round': server_round }
        fit_ins = FitIns(parameters, config)
        # return list of tuples of client and fit instructions
        # fit instructions are just global model parameters and config.
        return [ (client, fit_ins) for client in clients]

    # below works for clients sending one batch at a time
    def __convert_results_to_tensor(self,
        results: List[Tuple[ClientProxy, FitRes]],
        test=False
        ):
        numpy_input = np.empty((self.num_clients, self.batch_size, self.dim_input // self.num_clients))

        client_tensors = []
        clients_map = {}

        for i, (client, fit_response) in enumerate(results):
            # get client embeddings as numpy arrays
            client_embeddings = None
            if not test:
                client_embeddings = parameters_to_ndarrays(fit_response.parameters)
            else:
                client_embeddings = loads(fit_response.metrics['test_embeddings'])
            for j, embeds in enumerate(client_embeddings):
                numpy_input[i,j,:] = embeds.astype(np.float32)
            
            clients_map[client.cid] = i
            client_tensors.append(
                torch.tensor(numpy_input[i],dtype=torch.float32,requires_grad=True if not test else False)
            )

        gbl_model_input = torch.cat(client_tensors,1)

        if not test:
            self.client_tensors = client_tensors
            self.clients_map = clients_map

        return gbl_model_input

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        # convert all intermediate results from clients to 
        # input to the global model
        gbl_model_input = self.__convert_results_to_tensor(results=results)
        gbl_model_output = self.model(gbl_model_input)

        # upper idx
        # idx_u = None
        # # if the lower index is less than the number of labels
        # if self.train_lwr_idx < len(self.train_labels):
        #     # last batch has less than batch_size samples
        #     if len(self.train_labels) - self.train_lwr_idx < self.batch_size:
        #         idx_u = len(self.train_labels)
        #     else:
        #         # update upper bound normally
        #         idx_u = self.train_lwr_idx + self.batch_size
        # else:
        #     # just passed final batch
        #     self.train_lwr_idx = 0
        #     idx_u = self.train_lwr_idx + self.batch_size

        
        idx_u = self.update_label_index()

        self.optim.zero_grad()
        loss = self.criterion(
            gbl_model_output,
            self.train_labels[
                self.train_lwr_idx : idx_u
            ].float()
        )
        loss.backward()
        self.optim.step()

        preds = torch.round(gbl_model_output)
        num_correct = (
                preds == self.train_labels[self.train_lwr_idx : idx_u] 
            ).float().sum().squeeze().item()

        self.train_correct += num_correct
        

        self.train_lwr_idx += self.batch_size
        
        print(f'\ntrainAccuracy: {num_correct / len(gbl_model_input)}, \tnum samples: {len(gbl_model_input)}\n')
        return (ndarrays_to_parameters(
            get_parameters(self.model)
        ), {'loss': str(loss.item()), })

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:

        # sample all clients
        clients = client_manager.sample(
            num_clients=self.num_clients, min_num_clients=self.num_clients
        )
        # empty config
        config = {'server_round': server_round}
        # instructions for clients
        eval_ins = []
        
        for client in clients:
            idx = self.clients_map[client.cid]
            # provide gradient to clients
            tensor = self.client_tensors[idx]
            
            ins = EvaluateIns(
                ndarrays_to_parameters(tensor.grad.numpy()),
                config
            )
            eval_ins.append(ins)
        return [(client,eins) for (client,eins) in zip(clients,eval_ins)]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        criterion = BCELoss()
        
        gbl_model_input = self.__convert_results_to_tensor(results,test=True)
        
        with torch.no_grad():
            gbl_model_output = self.model(gbl_model_input).float()
            idx_u = self.update_label_index(test=True)
            
            loss = criterion(
                gbl_model_output,
                self.test_labels[
                    self.test_lwr_idx : idx_u
                ].float()
            )
            preds = torch.round(gbl_model_output)
            num_correct = (
                preds == self.test_labels[self.test_lwr_idx : idx_u] 
            ).float().sum().squeeze().item()

            acc = num_correct / len(gbl_model_input)
            f1 = f1_score(self.test_labels[self.test_lwr_idx : idx_u].numpy(), preds.numpy())

            self.test_f1_accum += f1
            self.test_lwr_idx += self.batch_size
            self.test_correct += num_correct
            
        return (server_round, {'loss': str(loss.item()), 'accuracy': acc, 'f1': f1})

    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the current model parameters."""
        # print(f"Strategy.evaluate ... server_round: {server_round}")
        pass

