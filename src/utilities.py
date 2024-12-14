### Copied https://github.com/locuslab/edge-of-stability/blob/github/src/utilities.py
from typing import List, Tuple, Iterable
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader, Subset
import os
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
import wandb
from zmq import device

# the default value for "physical batch size", which is the largest batch size that we try to put on the GPU
DEFAULT_PHYS_BS = 1000

# TODO: Add logs with the size of the matrix during calculation

# ======================================================
# Hessian Calculation Utilities
# ======================================================

def iterate_dataset(dataset: Dataset, batch_size: int, device = 'cpu'):
    """Iterate through a dataset, yielding batches of data."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch_X, batch_y in loader:
        yield batch_X.to(device), batch_y.to(device)

def compute_hvp(network: nn.Module, loss_fn: nn.Module,
                dataset: Dataset, vector: Tensor, physical_batch_size: int = DEFAULT_PHYS_BS, 
                device = 'cpu'):
    """Compute a Hessian-vector product."""
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    hvp = torch.zeros(p, dtype=torch.float, device=device)
    vector = vector.to(device)
    for (X, y) in iterate_dataset(dataset, physical_batch_size, device=device):
        loss = loss_fn(network(X), y) / n
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads)
    return hvp

def lanczos(matrix_vector, dim: int, neigs: int, device = 'cpu'):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        vec_tensor = torch.tensor(vec, dtype=torch.float).to(device)

        return matrix_vector(vec_tensor)

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()


def get_hessian_eigenvalues(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                            neigs=6, physical_batch_size=1000, device = 'cpu'):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hvp(network, 
                                          loss_fn, dataset, delta, 
                                          physical_batch_size=physical_batch_size, 
                                          device = device).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs, device = device)
    return evals, evecs.to(device)

# ======================================================
# Space Overlap Utilities
# ======================================================

def overlap_top_direct(top_evecs1, top_evecs2, device = "cpu"):
    """
        https://arxiv.org/abs/1812.04754 formula (5) using norm of projection.
        given two tensors shapes (k, d), k - dim of top space, d - dim of model.    
    """
    top_evecs1 = top_evecs1.to(device) 
    top_evecs2 = top_evecs2.to(device)
    assert len(top_evecs1.shape) == 2 and top_evecs1.shape == top_evecs2.shape
    k = top_evecs1.shape[0]

    dot_products = (top_evecs1 @ top_evecs2.T)
    projections =  (dot_products @ top_evecs1)
    frobenius_norm = torch.norm(projections, p = "fro")
    return torch.sum(frobenius_norm ** 2) / k

def overlap_top_tr(top_evecs1, top_evecs2, device="cpu"):
    """
        https://arxiv.org/abs/1812.04754 formula (5) the trace of projection.
        given two tensors shapes (k, d), k - dim of top space, d - dim of model.    
    """
    top_evecs1 = top_evecs1.to(device) 
    top_evecs2 = top_evecs2.to(device)
    assert len(top_evecs1.shape) == 2 and top_evecs1.shape == top_evecs2.shape
    k = top_evecs1.shape[0]

    tr_top1 = torch.sum(top_evecs1 * top_evecs1)  # Tr(P_t)
    tr_top2 = torch.sum(top_evecs2 * top_evecs2)  # Tr(P'_t)
    tr_composition = torch.sum((top_evecs1 @ top_evecs2.T) ** 2)  # Tr(P_t P'_t)

    return tr_composition / torch.sqrt(tr_top1 * tr_top2)

def overlap_bulk_tr(top_evecs1, top_evecs2, device="cpu"):
    """
        https://arxiv.org/abs/1812.04754 formula (5) the trace of projection.
        given two tensors shapes (k, d), k - dim of top space, d - dim of model.    
    """
    top_evecs1 = top_evecs1.to(device)
    top_evecs2 = top_evecs2.to(device)
    assert len(top_evecs1.shape) == 2 and top_evecs1.shape == top_evecs2.shape
    k = top_evecs1.shape[0]
    d = top_evecs2.shape[1]
    d_tensor = torch.tensor(d, dtype=float)

    tr_top1 = torch.sum(top_evecs1 * top_evecs1)  # Tr(P_t)
    tr_top2 = torch.sum(top_evecs2 * top_evecs2)  # Tr(P'_t)
    tr_composition = torch.sum((top_evecs1 @ top_evecs2.T) ** 2)  # Tr(P_t P'_t)

    return (d_tensor - tr_top1 - tr_top2 + tr_composition) / torch.sqrt((d_tensor - tr_top1) * (d_tensor - tr_top2))

# ======================================================
# Deprecated: Projected Step Utils
# ======================================================
    

def projected_step_bulk(model, loss, criterion, dataset, batch_size, lr, device):
    evals, evecs = get_hessian_eigenvalues(model, criterion, dataset,
                                           physical_batch_size=batch_size,
                                           neigs=10, device=device)
    evecs.transpose_(1, 0)

    with torch.no_grad():
        grad = torch.autograd.grad(
            loss, inputs=model.parameters(), create_graph=True)
        vec_grad = parameters_to_vector(grad)
        step = vec_grad.detach()
        for vec in evecs:
            step -= torch.dot(vec_grad, vec) * vec
        vec_params = parameters_to_vector(model.parameters())
        vec_params -= step * lr
        vector_to_parameters(vec_params, model.parameters())
        model.zero_grad()


def projected_step_top(model, loss, criterion, dataset, batch_size, lr, device):
    evals, evecs = get_hessian_eigenvalues(model, criterion, dataset,
                                           physical_batch_size=batch_size,
                                           neigs=10, device=device)
    evecs.to(device).transpose_(1, 0)

    with torch.no_grad():  # Ensure we don’t track these operations for gradient computation
        grad = torch.autograd.grad(
            loss, inputs=model.parameters(), create_graph=True)
        vec_grad = parameters_to_vector(grad).to(device)
        step = torch.Tensor(vec_grad.shape).to(device)
        for vec in evecs:
            step += torch.dot(vec_grad, vec) * vec
        vec_params = parameters_to_vector(model.parameters())
        vec_params -= step * lr
        vector_to_parameters(vec_params, model.parameters())
        model.zero_grad()


def projected_step_bulk_of_prev_exp(model, evals, evecs, loss, criterion, dataset, batch_size, lr, device):
    with torch.no_grad():
        grad = torch.autograd.grad(
            loss, inputs=model.parameters(), create_graph=True)
        vec_grad = parameters_to_vector(grad)
        step = vec_grad.detach()
        for vec in evecs:
            step -= torch.dot(vec_grad, vec) * vec
        vec_params = parameters_to_vector(model.parameters())
        vec_params -= step * lr
        vector_to_parameters(vec_params, model.parameters())
        model.zero_grad()

# ======================================================
# Time Measurement Utilities
# ======================================================

def timeit(func):
    """Decorator that times the execution of a function."""
    def timed(*args, **kwargs):
        start_time = time.time()  
        result = func(*args, **kwargs)  
        end_time = time.time() 
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result
    
    return timed

@contextmanager
def time_block(label="Code block"):
    """Context manager to time a specific block of code."""
    start_time = time.time()
    try:
        yield  # Allow the block of code to execute
    finally:
        end_time = time.time()
        print(f"{label} executed in {end_time - start_time:.4f} seconds")


# ======================================================
# Result Saving Utilities
# ======================================================

def save_results(args, warm_up_losses, train_losses, warm_up_accuracy, final_accuracy, warm_up_training_steps):
    per_batch_losses = warm_up_losses[0] + train_losses[0]
    per_epoch_losses = warm_up_losses[1] + train_losses[1]
    training_steps_per_batch = warm_up_losses[2] + train_losses[2]
    training_steps_per_epoch = warm_up_losses[3] + train_losses[3]
    output_name = f"{args.dataset}_{args.task}_{args.model}_{args.activation}_{args.algo}"    
    if args.hidden_sizes:
        output_name += "_hidden_sizes_" + "-".join(map(str, args.hidden_sizes))
    if args.hidden_sizes:
        output_name += "_hid_sizes_" + "-".join(map(str, args.hidden_sizes))

    if args.plot_losses:
        plt.figure(1)
        plt.axvline(x=warm_up_training_steps, color='red', linestyle='--', label="End of warm-up")
        plt.plot(training_steps_per_batch, np.log(per_batch_losses))
        plt.title('Per Batch Losses (log-scale)')
        plt.xlabel('Training steps')
        plt.legend()

        plt.figure(2)
        plt.axvline(x=warm_up_training_steps, color='red', linestyle='--', label="End of warm-up")
        plt.plot(training_steps_per_epoch, np.log(per_epoch_losses))
        plt.title('Per Epoch Losses (log-scale)')
        plt.xlabel('Training steps')
        plt.legend()

        plt.show()
        output_dir = "../plots"

        plot_name = output_name+ '.png'
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, plot_name)
        plt.savefig(save_path)

    results = {}
    results['args'] = str(vars(args))
    results['warm_up_losses_batch'] = warm_up_losses[0]
    results['warm_up_losses_epoch'] = warm_up_losses[1]
    results['train_losses_batch'] = train_losses[0]
    results['train_losses_epoch'] = train_losses[1]
    results['warm_up_accurary'] = warm_up_accuracy
    results['final_accuracy'] = final_accuracy

    if args.save_results and not args.debug:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name_pickle = os.path.join(args.storage, f"projected_training_{current_time}_{output_name}.pkl")
        with open(file_name_pickle, "wb") as file:
            pickle.dump(results, file)
        file_name_json = os.path.join(args.storage, f"projected_training_{current_time}_{output_name}.json")
        with open(file_name_json, 'w') as json_file:
            json.dump(results, json_file, indent=4)  
        print(f'Results saved in {file_name_pickle, file_name_json}!')

        # saving to wandb
        artifact = wandb.Artifact(name = "results", type = "dict")
        artifact.add_file(local_path = file_name_pickle, name = "results_pickle")
        artifact.add_file(local_path = file_name_json, name = "results_json")
        artifact.save()

def save_results_cl(args, all_warm_up_losses, all_train_losses, final_accuracy=0):
    num_experiences = len(all_warm_up_losses)
    fig, axes = plt.subplots(2, num_experiences, figsize=(
        5 * num_experiences, 4), constrained_layout=True)
    if num_experiences == 1:
        axes = [[axes[0]], [axes[1]]]
    output_name = f"CL_{args.dataset}_{args.task}_{args.model}_{args.activation}_{args.algo}_seed_{args.seed}"
    if args.hidden_sizes:
        output_name += "_hid_sizes_" + "-".join(map(str, args.hidden_sizes))

        for i, (warm_up_losses, train_losses) in enumerate(zip(all_warm_up_losses, all_train_losses)):
            # Combine batch and epoch losses
            per_batch_losses = warm_up_losses[0] + train_losses[0]
            per_epoch_losses = warm_up_losses[1] + train_losses[1]

            # Combine training steps
            training_steps_per_batch = warm_up_losses[2] + train_losses[2]
            training_steps_per_epoch = warm_up_losses[3] + train_losses[3]

            # Determine end of warm-up step
            warm_up_training_steps = 0 if len(
                warm_up_losses[2]) == 0 else warm_up_losses[2][-1]

            # First row: Per Batch Losses
            axes[0][i].axvline(x=warm_up_training_steps, color='red',
                               linestyle='--', label="End of warm-up")
            axes[0][i].plot(training_steps_per_batch, np.log1p(
                per_batch_losses), label=f"Experience {i + 1}")
            axes[0][i].set_title(f"Per Batch Losses (Exp {i + 1})")
            axes[0][i].set_xlabel("Training Step")
            axes[0][i].set_ylabel("Log Loss")
            axes[0][i].grid(True)
            axes[0][i].legend()

            # Second row: Per Epoch Losses
            axes[1][i].axvline(x=warm_up_training_steps, color='red',
                               linestyle='--', label="End of warm-up")
            axes[1][i].plot(training_steps_per_epoch, np.log1p(
                per_epoch_losses), label=f"Experience {i + 1}")
            axes[1][i].set_title(f"Per Epoch Losses (Exp {i + 1})")
            axes[1][i].set_xlabel("Training Step")
            axes[1][i].set_ylabel("Log Loss")
            axes[1][i].grid(True)
            axes[1][i].legend()

    plt.suptitle("Losses (log scale)")
    plt.show()
    output_dir = "../plots"
    plot_name = output_name + '.png'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, plot_name)
    plt.savefig(save_path)

    results = {}
    results['args'] = args
    results['all_train_losses'] = all_train_losses
    results['all_warm_up_losses'] = all_warm_up_losses
    results['final_accuracy'] = final_accuracy

    if args.save_results and not args.debug:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name_pickle = os.path.join(args.storage, f"cl_tast_{current_time}_{output_name}.pkl")
        with open(file_name_pickle, "wb") as file:
            pickle.dump(results, file)
        file_name_json = os.path.join(args.storage, f"cl_task_{current_time}_{output_name}.json")
        with open(file_name_json, 'w') as json_file:
            json.dump(results, json_file, indent=4)  
        print(f'Results saved in {file_name_pickle, file_name_json}!')

        # saving to wandb
        artifact = wandb.Artifact(name = "results", type = "dict")
        artifact.add_file(local_path = file_name_pickle, name = "results_pickle")
        artifact.add_file(local_path = file_name_json, name = "results_json")
        artifact.save()

