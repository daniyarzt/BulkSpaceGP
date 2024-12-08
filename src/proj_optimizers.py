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

    def step(self, model, criterion, dataset):
        for group in self.param_groups:
            params: List[torch.Tensor] = []
            grads: List[torch.Tensor] = []
            self._init_group(group, params, grads)
            vec_params = torch.cat(params).to(self.device)
            vec_grad = torch.cat(grads).to(self.device)
            
            _, evecs = get_hessian_eigenvalues(model, criterion, dataset,
                                               physical_batch_size=self.batch_size, 
                                               neigs=10, device=self.device)
            
            evecs.transpose_(1, 0)

            step = vec_grad.detach() 
            with torch.no_grad():
                for vec in evecs:
                    step -= torch.dot(step, vec) * vec
                vec_params.add_(step, alpha=-group['lr'])
                vector_to_parameters(vec_params, group["params"])


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

    def step(self, model, criterion, dataset):
        for group in self.param_groups:
            params: List[torch.Tensor] = []
            grads: List[torch.Tensor] = []
            self._init_group(group, params, grads)
            vec_params = torch.cat(params).to(self.device)
            vec_grad = torch.cat(grads).to(self.device)
            
            _, evecs = get_hessian_eigenvalues(model, criterion, dataset,
                                               physical_batch_size=self.batch_size, 
                                               neigs=10, device=self.device)
            
            evecs.transpose_(1, 0)

            step = torch.Tensor(vec_grad.shape).to(self.device)
            with torch.no_grad():
                for vec in evecs:
                    step += torch.dot(step, vec) * vec
                vec_params.add_(step, alpha=-group['lr'])
                vector_to_parameters(vec_params, group["params"])