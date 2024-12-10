import torch
from utilities import get_hessian_eigenvalues
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from typing import List, Union


class BulkSGD(torch.optim.Optimizer):
    
    def __init__(self, params, batch_size, device,
                 lr: Union[float, torch.Tensor] = 1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.batch_size = batch_size
        self.device = device

    def _init_group(self, group, params, grads):
        for p in group["params"]:
            if p.grad is not None:
                params.append(p.data.view(-1))
                grads.append(p.grad.data.view(-1))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            vec_params = parameters_to_vector([p for p in group["params"] if p.grad is not None]).to(self.device)
            vec_grad = parameters_to_vector([p.grad for p in group["params"] if p.grad is not None]).to(self.device)
            projections = (self.evecs @ vec_grad.unsqueeze(1)).squeeze(1)
            vec_grad -= (projections.unsqueeze(0) @ self.evecs).squeeze(0)
            vec_params.add_(vec_grad, alpha=-group['lr'])
            vector_to_parameters(vec_params, group["params"])
    
    def calculate_evecs(self, model, criterion, dataset):
        with torch.enable_grad():
                _, self.evecs = get_hessian_eigenvalues(model, criterion, dataset,
                                                   physical_batch_size=self.batch_size,
                                                   neigs=10, device=self.device)
                self.evecs.transpose_(1, 0)


class TopSGD(torch.optim.Optimizer):
    
    def __init__(self, params, batch_size, device,
                 lr: Union[float, torch.Tensor] = 1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.batch_size = batch_size
        self.device = device

    def _init_group(self, group, params, grads):
        for p in group["params"]:
            if p.grad is not None:
                params.append(p.data.view(-1))
                grads.append(p.grad.data.view(-1))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            vec_params = parameters_to_vector([p for p in group["params"] if p.grad is not None]).to(self.device)
            vec_grad = parameters_to_vector([p.grad for p in group["params"] if p.grad is not None]).to(self.device)
            projections = (self.evecs @ vec_grad.unsqueeze(1)).squeeze(1)
            vec_grad = (projections.unsqueeze(0) @ self.evecs).squeeze(0)
            vec_params.add_(vec_grad, alpha=-group['lr'])
            vector_to_parameters(vec_params, group["params"])

    def calculate_evecs(self, model, criterion, dataset):
        with torch.enable_grad():
            _, self.evecs = get_hessian_eigenvalues(model, criterion, dataset,
                                                    physical_batch_size=self.batch_size,
                                                    neigs=10, device=self.device)
        self.evecs.transpose_(1, 0)