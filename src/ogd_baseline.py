import torch
from torch.utils.data import DataLoader, Subset
from avalanche.core import SupervisedPlugin
import random


class OGDPlugin(SupervisedPlugin):
    """
    Orthogonal Gradient Descent (OGD) Plugin for Avalanche.
    Implements gradient projection to avoid catastrophic forgetting.
    """
    def __init__(self, n_subset=200):
        """
        Initialize the OGD plugin.

        Args:
            n_subset (int): Number of gradient directions to store per task.
        """
        super().__init__()
        self.S = []  # Stores gradient directions
        self.n_subset = n_subset

    def _clone_grads(self, params):
        return [param.grad.clone() for param in params if param.grad is not None]

    def _proj(self, v, g):

        # Calculate the projection factor
        v_flatten = torch.nn.utils.parameters_to_vector(v)
        g_flatten = torch.nn.utils.parameters_to_vector(g)
        scale = torch.dot(v_flatten, g_flatten) / torch.dot(v_flatten, v_flatten)
        
        # Calculate the projection (while keeping the shape)
        return [scale * param for param in v]

    def _store_gradient(self, grad):
        self.S.append(grad)

    def before_update(self, strategy, **kwargs):
        """
        Hook called before the parameters are updated.
        """

        # Stochastic/Batch Gradient for Tk at w
        g = self._clone_grads(strategy.model.parameters())
        
        # Projections
        projs = [self._proj(v, g) for v in self.S]

        #print("Projecting gradients...")
        # g* = g − ∑v∈S proj_v(g)
        for param_grads in projs:
            for idx, param_grad in enumerate(param_grads):
                g[idx] -= param_grad 
        
        #print("Updating model gradients...")
        # w* ← w - ηg* (i.e, just setting the gradients in the model to the projected ones)
        for idx, param in enumerate(strategy.model.parameters()):
            if param.grad is not None:
                param.grad.copy_(g[idx])

    def after_training_exp(self, strategy, **kwargs):
        """
        Hook called after completing training on a task (experience).
        """

        # Get a subset of data set

        dataset_size = len(strategy.experience.dataset)
        random_indices = torch.randperm(dataset_size)[:self.n_subset].tolist()
        subset = Subset(strategy.experience.dataset, random_indices)

        loader = DataLoader(subset, batch_size=1, shuffle=False)
        
        # for (x, y) ∈ T_t and k ∈ [1, c] s.t. yk = 1 do
        for x, y, *_ in loader:
            x, y = x.to(strategy.device), y.to(strategy.device)

            # Calculate ∇f_k(x; w)
            strategy.model.zero_grad()
            logits = strategy.model(x)
            logit_k = logits[0, y]
            logit_k.backward()
            
            u = self._clone_grads(strategy.model.parameters())

            # Projections
            projs = [self._proj(v, u) for v in self.S]

            # u ← ∇f_k(x; w) - ∑v∈S proj_v(∇_fk(x; w))
            for param_grads in projs:
                for idx, param_grad in enumerate(param_grads):
                    u[idx] -= param_grad

            # S ← S ∪ {u}
            self._store_gradient(u)
        strategy.model.zero_grad()
        print("Gradients stored.")