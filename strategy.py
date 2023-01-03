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


from torch import optim
from torch.nn import BCELoss
import torch

torch.manual_seed(0)
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

class SplitVFL(Strategy):
    def __init__(self, num_clients, batch_size, dim_input, labels, lr=1e-6):
        super(SplitVFL, self).__init__()

        self.batch_size = batch_size
        self.num_clients = num_clients
        self.dim_input = dim_input
        self.lr = lr
        self.criterion = BCELoss()
        self.labels = torch.tensor(labels)

        self.model = None
        self.client_tensors = None
        self.clients_map = None
        self.optim = None
        
    def __repr__(self) -> str:
        return "VerticalFed"
    
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
        # fit instructions are just global mode parameters and config.
        return [ (client, fit_ins) for client in clients]
    
    def __convert_results_to_tensor(self,
        results: List[Tuple[ClientProxy, FitRes]],
        ):
        numpy_input = np.empty((self.num_clients, self.batch_size, self.dim_input // self.num_clients))

        client_tensors = []
        clients_map = {}

        for i, (client, fit_response) in enumerate(results):
            # get client embeddings as numpy arrays
            client_embeddings = parameters_to_ndarrays(fit_response.parameters)
            for j, embeds in enumerate(client_embeddings):
                numpy_input[i,j,:] = embeds.astype(np.float32)
            
            clients_map[client.cid] = i
            client_tensors.append(
                torch.tensor(numpy_input[i],dtype=torch.float32,requires_grad=True)
            )

        gbl_model_input = torch.cat(client_tensors,1)

        self.client_tensors = client_tensors
        self.clients_map = clients_map

        return gbl_model_input

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        
        gbl_model_input = self.__convert_results_to_tensor(results=results)
        gbl_model_output = self.model(gbl_model_input)
        idx = self.batch_size * server_round
        self.optim.zero_grad()
        loss = self.criterion(
            gbl_model_output,
            self.labels[idx:idx+self.batch_size].float()
        )
        loss.backward()
        self.optim.step()
        num_correct = (
                torch.round(gbl_model_output) == self.labels[idx:idx+self.batch_size] 
            ).float().sum().squeeze().item()
        print(f"\n\n\n\n{num_correct}\n\n\n")
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
        # print("\n\n\nAGGREGATE_EVALUATE\n\n\n")
        return (None, {'None': str(None)})

    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the current model parameters."""
        # print(f"Strategy.evaluate ... server_round: {server_round}")
        pass
    # def results_to_tensor(self, results):
    #     batch_size = self.batch_size if self.is_training_mode else self.test_length
        
    #     input_numpy = np.empty((self.num_clients, batch_size, self.dim_input // self.num_clients))
    #     client_tensors = []
    #     clients_map = dict()

    #     for i, (client, fit_response) in enumerate(results):
    #         params = fit_response.parameters
    #         # Convert to ndarray
    #         weights_per_batch = parameters_to_ndarrays(params)
    #         for j, arr in enumerate(weights_per_batch):
    #             # (client, batch, weight)
    #             input_numpy[i, j, :] = arr.astype(np.float32)
    #         clients_map[client.cid] = i
    #         client_tensors.append(
    #             torch.tensor(input_numpy[i,:,:], dtype=torch.float32, requires_grad=True)
    #         )

    #     input_tensor = torch.cat(client_tensors, 1)
        
    #     # remember clients for evaluation
    #     self.client_tensors = client_tensors
    #     self.clients_map = clients_map

    #     return input_tensor

