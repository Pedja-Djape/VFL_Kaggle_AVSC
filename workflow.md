# Vertical Federated Learning with Flower (Simulation)

This document outlines the workflow and steps required to
train a global model in a vertical learning setting.

## Preliminary Steps

1. Acquire data for training (i.e., features and labels)
2. Separate features into K sets and get labels.
    1. Labels will be held with active party 
    2. Active party will host the global mega-model and 
    will not have training data of it's own. 

## Training Workflow

1. Flower Server calls our custom strategy for initial 
parameters.
    1. Strategy will then instantiate an instance of the global model. **Important:** When creating the model 
    the strategy has to be aware of the total number of features coming from the clients. I.e., $D$ = $\sum_{i=1}^K d_i$, where $d_i$ is the output of the local model of client $i$
2. Once the Server has obtained parameters from the custom strategy, the server once again calls upon the Strategy but this time for instructions to provide clients. Specifically, the Server calls the Strategy's `configure_fit` function. 
    
