import pytest
from train import * 
from utilities import *

import torch 
from torch import nn, functional
from torch.nn import Linear
from torch.utils.data import TensorDataset

EPS = 1e-6

def test_compute_hvp_with_chain_nn(): 

    class LinearNN(nn.Module):
        def __init__(self):
            super(LinearNN, self).__init__()
            
            self.fn = Linear(3, 1, bias = False)
            self.fn.weight = nn.Parameter(torch.tensor([1., 2., 3.]))

        def forward(self, x):
            return self.fn(x)

    x = torch.tensor([[1., 2., 3.], [1., 2., 3.]])
    y = torch.tensor([15., 15.])
    dataset = TensorDataset(x, y)
    criterium = nn.MSELoss()
    model = LinearNN()

    vector = torch.tensor([1., 2., 3.])
    result = compute_hvp(model, criterium, dataset, vector, 1)
    assert np.max(np.asarray(result) - np.array([28., 56., 84.])) < EPS

def test_autograd_grad_doesnt_change_param_grads(): 

    class LinearNN(nn.Module):
        def __init__(self):
            super(LinearNN, self).__init__()
            
            self.fn = Linear(3, 1, bias = False)
            self.fn.weight = nn.Parameter(torch.tensor([1., 2., 3.]))

        def forward(self, x):
            return self.fn(x)

    x = torch.tensor([[1., 2., 3.], [1., 2., 3.]])
    y = torch.tensor([15., 15.])
    dataset = TensorDataset(x, y)
    criterium = nn.MSELoss()
    model = LinearNN()

    y_ = model.forward(x)
    current_loss = criterium(y_, y)

    grads = torch.autograd.grad(current_loss, model.parameters())

    for param in model.parameters():
        assert param.grad is None
