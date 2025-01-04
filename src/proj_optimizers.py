import torch
from utilities import get_hessian_eigenvalues, overlap_top_tr
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from typing import List, Union
import wandb


class BulkSGD(torch.optim.SGD):
    
    def __init__(self, params, batch_size, device,
                 lr: Union[float, torch.Tensor] = 1e-3):
        super().__init__(params, lr)
        self.batch_size = batch_size
        self.device = device
        self._warm_up = True

    @torch.no_grad()
    def step(self):
        if self._warm_up:
            super().step()
            return; 
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


class TopSGD(torch.optim.SGD):
    
    def __init__(self, params, batch_size, device,
                 lr: Union[float, torch.Tensor] = 1e-3):
        super().__init__(params, lr)
        self.batch_size = batch_size
        self.device = device
        self._warm_up = True

    @torch.no_grad()
    def step(self):
        if self._warm_up:
            super().step()
            return; 
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

class CLBulkSGD(torch.optim.SGD): 

    def __init__(self, params, batch_size, device,
                 lr: Union[float, torch.Tensor] = 1e-3, mode = 'average', n_evecs = 10):
        super().__init__(params, lr)
        self.batch_size = batch_size
        self.device = device
        self._warm_up = True
        self.top_evecs = []
        self._log_overlaps = False
        self.mode = mode 
        print(f'Inintialized CLBulkSGD optimizer with {mode} mode, and n_evecs = {n_evecs}')
        if mode not in ['average', 'gs', 'only_first', 'only_last']:
            raise NotImplementedError()
        self.n_evecs = n_evecs

    @torch.no_grad()
    def step(self):
        if self._warm_up or self.top_evecs is None:
            super().step()
            return
        k = len(self.top_evecs)
        alpha = 1.0 / k if self.mode == 'average' else 1.
        top_space_projection = torch.concatenate(self.top_evecs, dim = 0)
        for group in self.param_groups:
            vec_params = parameters_to_vector([p for p in group["params"] if p.grad is not None]).to(self.device)
            vec_grad = parameters_to_vector([p.grad for p in group["params"] if p.grad is not None]).to(self.device)
            projections = (top_space_projection @ vec_grad.unsqueeze(1)).squeeze(1)
            vec_grad -= (projections.unsqueeze(0) @ top_space_projection).squeeze(0) * alpha
            vec_params.add_(vec_grad, alpha=-group['lr'])
            vector_to_parameters(vec_params, group["params"])
    
    def calculate_evecs(self, model, criterion, dataset):
        with torch.enable_grad():
                _, evecs = get_hessian_eigenvalues(model, criterion, dataset,
                                                   physical_batch_size=self.batch_size,
                                                   neigs=self.n_evecs, device=self.device)
                evecs.transpose_(1, 0)
                self.top_evecs = [evecs]

    def gram_schmidt(self, evecs):
        for evec in evecs:
            for top_space in self.top_evecs:
                    top_space_projection = torch.concatenate(self.top_evecs, dim = 0)
                    projection_mag = (top_space_projection @ evec.unsqueeze(1)).squeeze(1) 
                    evec -= (projection_mag.unsqueeze(0) @ top_space_projection).squeeze(0)
        return evecs

    def append_evecs(self, model, criterion, dataset):
        with torch.enable_grad():
                _, evecs = get_hessian_eigenvalues(model, criterion, dataset,
                                                   physical_batch_size=self.batch_size,
                                                   neigs=self.n_evecs, device=self.device)
                
                evecs.transpose_(1, 0)

                if self._log_overlaps and len(self.top_evecs) > 0:
                     for past_evecs in self.top_evecs:
                          overlap = overlap_top_tr(past_evecs, evecs, self.device)
                          print(f'top overlap = {overlap}')
                          wandb.log({'top_overlap' : overlap})
                
                if self.mode == 'gs':
                    self.top_evecs.append(self.gram_schmidt(evecs))
                elif self.mode == 'average':
                    self.top_evecs.append(evecs)
                elif self.mode == 'only_first':
                    self.top_evecs = [evecs] if len(self.top_evecs) == 0 else self.top_evecs
                elif self.mode == 'only_last':
                    self.top_evecs = [evecs]