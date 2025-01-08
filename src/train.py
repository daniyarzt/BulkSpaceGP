import os 
import pathlib
import random
from argparse import ArgumentParser, BooleanOptionalAction

from utilities import get_hessian_eigenvalues, timeit, time_block, save_results, save_results_cl, get_projected_sharpness, WandBAccuracyLogger
from proj_optimizers import BulkSGD, TopSGD, CLBulkSGD
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np
from torchsummary import summary

import wandb
from avalanche.benchmarks import PermutedMNIST
from avalanche.training.templates import SupervisedTemplate

from avalanche.evaluation.metrics import MinibatchLoss, EpochLoss, TaskAwareLoss, StreamAccuracy
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

from avalanche.training.supervised import EWC, AGEM
from gpm_baseline import GPM
from collections import OrderedDict

DEVICE = 'cpu'

TOP_EVECS = []
TOP_EVEC_RECORD_FREQ = 1
TOP_EVEC_TIMER = 0
SHARPNESS_TIMER = 0
SHARPNESS_FREQ = 25
baselines = ["ewc", "agem", "ogd", "gpm"]

def arg_parser():
    parser = ArgumentParser(description='Train')

    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--debug', action=BooleanOptionalAction, default=False)
    parser.add_argument('--storage', type=pathlib.Path, default=os.path.join("..", "storage"))
    parser.add_argument('--save_results', action=BooleanOptionalAction, default=True)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--epochs', type=int, default=5, required=False)
    parser.add_argument('--dataset', choices = ['MNIST_5k', 'MNIST_full'], default='MNIST_5k')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--loss', type=str, default='cross_entropy_loss')

    # Model parameters 
    parser.add_argument('--model', choices = ['MLP'], default='MLP')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', required=False)
    parser.add_argument('--activation', choices = ['relu','tanh'], default='relu')

    # projected_training args
    parser.add_argument('--warm_up_epochs', type=int, default=0, required=False)
    parser.add_argument('--algo', choices=['SGD', 'Bulk-SGD', 'Top-SGD', 'prev_Bulk-SGD'] + baselines, default='SGD')
    parser.add_argument('--plot_losses', action=BooleanOptionalAction, default=False)

    # CL task 
    parser.add_argument('--n_experiences', type=int, default=2, required=False)
    parser.add_argument('--n_bulk_batches', type=int, default=1)
    parser.add_argument('--hessian_subset_size', type=int, default=1000)
    parser.add_argument('--mode', choices=['average', 'gs', 'only_first', 'only_last'], default='gs')
    parser.add_argument('--n_evecs', type=int, default=10)

    # Additional logs and metrics 
    parser.add_argument('--save_evecs', action=BooleanOptionalAction, default=False)
    parser.add_argument('--plot_sharpness', action=BooleanOptionalAction, default=False)
    parser.add_argument('--log_overlaps', action=BooleanOptionalAction, default=False)
    parser.add_argument('--holdout_acc_freq', type=int, default=1)

    args = parser.parse_args()
    return args

def train_avalanche(args, strategy, benchmark):
    # Train and evaluate
    for experience in benchmark.train_stream:
        print(f"Start training on experience {experience.current_experience}")
        strategy.train(experience)
    training_statistics = strategy.evaluator.get_all_metrics()
    training_steps_per_batch, per_batch_losses = training_statistics["Loss_MB/train_phase/train_stream/Task000"]
    training_steps_per_epoch, per_epoch_losses = training_statistics["Loss_Epoch/train_phase/train_stream/Task000"]

    print(training_statistics)

    epoch_size = len(training_steps_per_batch) // (args.n_experiences * args.epochs)

    all_train_losses = [(per_batch_losses, per_epoch_losses, [x*args.batch_size for x in training_steps_per_batch], [x*args.batch_size*epoch_size for x in training_steps_per_epoch])]
    final_accuracy = strategy.eval(benchmark.test_stream)
    return final_accuracy, all_train_losses


def run_avalanche(args, strategy_name, hyperparamters, model, optimizer, criterion, benchmark, holdout_datasets):
    strategies = {
        "ewc": EWC,
        "agem": AGEM,
        "gpm" : GPM
    }
    
    eval_plugin = EvaluationPlugin(
        MinibatchLoss(),
        EpochLoss(),
        StreamAccuracy(),
        loggers=[InteractiveLogger()]
    )
    wandb_acc_logger = WandBAccuracyLogger(holdout_datasets, args.batch_size, args.holdout_acc_freq)

    strategy = strategies[strategy_name](
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        **hyperparamters,
        evaluator=eval_plugin, 
        plugins = [wandb_acc_logger]
    )
    
    result =  train_avalanche(args, strategy, benchmark)

    metrics = eval_plugin.get_all_metrics()
    
    print(f'Average accuracy: {metrics["Top1_Acc_Stream/eval_phase/test_stream/Task000"][1][0] * 100.}')
    wandb.log({'ACC' : metrics["Top1_Acc_Stream/eval_phase/test_stream/Task000"][1][0] * 100.})

    return result

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(input_size : int, output_size : int, args, device, bias=True) -> nn.Module:
    # Model definitions could be moved to a separate file...  
    if args.model == 'MLP':
        # Define the MLP model 
        class MLP(nn.Module):
            def __init__(self, input_size, output_size, hidden_sizes, bias):
                super(MLP, self).__init__()
                # Needed for GPM 
                self.act=OrderedDict()
                self.n_lin = len(hidden_sizes) + 1
                
                layers = []
                in_size = input_size 

                for layer_id, hidden_size in enumerate(hidden_sizes):
                    if layer_id > 0:
                        if args.activation == 'relu':
                            layers.append(nn.ReLU())
                        elif args.activation == 'tanh':
                            layers.append(nn.Tanh())
                    layers.append(nn.Linear(in_size, hidden_size, bias=bias))
                    in_size = hidden_size
                
                layers.append(nn.Linear(in_size, output_size, bias=bias).to(device))
                self.model = nn.Sequential(*layers)
                
            def forward(self, x):
                x = x.view(-1, 28 * 28)
                i = 1
                for layer in self.model:
                    if isinstance(layer, nn.Linear):
                        self.act[f'Lin{i}'] = x
                        i += 1
                    x = layer(x)
                return x
            
        return MLP(input_size, output_size, args.hidden_sizes, bias)
    else:
        raise NotImplementedError()
    
def get_optimizer(args, model):
    lr = args.lr
    batch_size = args.batch_size

    if args.algo in ["SGD", "ewc", "agem", "gpm"]:
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif args.algo == "Bulk-SGD":
        optimizer = BulkSGD(model.parameters(), lr=lr, batch_size=batch_size, device=DEVICE)
    elif args.algo == "Top-SGD":
        optimizer = TopSGD(model.parameters(), lr=lr, batch_size=batch_size, device=DEVICE)
    elif args.algo == "prev_Bulk-SGD":
        optimizer = CLBulkSGD(model.parameters(), lr=lr, batch_size=batch_size, device=DEVICE, mode=args.mode, n_evecs=args.n_evecs)
    else:
        raise NotImplementedError()
    return optimizer

# TODO: Refactoring: instead of passing subset size, pass the subset itself. Then it will be consistent. 
# Right now we are reshuffling the dataset everytime we want to get a dataloader. 
def get_partial_dataloader(dataset, subset_size, batch_size):
    if subset_size is not None:
        subset_indices = np.random.choice(len(dataset), subset_size, replace=False)
        partial_dataset = Subset(dataset, subset_indices)
    else:
        partial_dataset = dataset
    return DataLoader(dataset=partial_dataset, batch_size=batch_size,
                                    shuffle=True, pin_memory=True)

def get_holdout_dataset(test_dataset, subset_size):
    subset_indices = np.random.choice(len(test_dataset), subset_size, replace = False)
    partial_dataset = Subset(test_dataset, subset_indices)
    return partial_dataset

def train(train_loader, model, criterion, optimizer, lr, num_epochs: int, algo: str = 'SGD', 
          cur_training_steps=0, num_classes=10, evals=[], evecs=[], phase = "", args = None, **kwargs):
    global TOP_EVEC_TIMER, TOP_EVEC_RECORD_FREQ, TOP_EVECS, SHARPNESS_FREQ, SHARPNESS_TIMER
    holdout_acc_timer = 0
    holdout_acc_freq = 1
    if args is not None:
        holdout_acc_freq = args.holdout_acc_freq
    
    save_evecs = False
    if args is not None and args.save_evecs:
        save_evecs = True

    losses = []
    per_epoch_losses = []
    training_steps_per_batch = []
    training_steps_per_epoch = []
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        loss_sum = 0.0
        batches = 0.0
        training_steps_per_epoch.append(cur_training_steps)
        with time_block("Training epoch"):
            for batch in tqdm(train_loader, "training loop..."):
                images, labels, *rest = batch
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                if isinstance(criterion, nn.MSELoss):
                    target_one_hot = F.one_hot(
                        labels, num_classes=num_classes).float().to(DEVICE)
                    labels = target_one_hot

                k = len(images)

                # Forward pass
                outputs = model(images).to(DEVICE)
                loss = criterion(outputs, labels)
                loss_sum += loss.item()
                batches += 1
                losses.append(loss.item())
                training_steps_per_batch.append(cur_training_steps)
                wandb.log({'loss_mb': loss.item(),
                          'training_step': cur_training_steps})

                cur_training_steps += k

                # Bit ugly sorry. To prevent double evec calculation.
                if args.save_evecs and algo == 'SGD':
                    if TOP_EVEC_TIMER % TOP_EVEC_RECORD_FREQ == 0:
                        dataset = TensorDataset(images, labels)
                        _, cur_evecs = get_hessian_eigenvalues(model, criterion, dataset,
                                                    physical_batch_size=len(images),
                                                    neigs=10, device=DEVICE)
                        cur_evecs.transpose_(1, 0)
                        TOP_EVECS.append((cur_training_steps, phase, cur_evecs.clone()))
                    TOP_EVEC_TIMER += 1

                # Plotting sharpness 
                if args.plot_sharpness:
                    if SHARPNESS_TIMER % SHARPNESS_FREQ == 0:
                        dataset = TensorDataset(images, labels)
                        top_evecs = [] if not hasattr(optimizer, 'top_evecs') else optimizer.top_evecs
                        sharpness = get_projected_sharpness(model, criterion, 
                                                            dataset, physical_batch_size=len(images),
                                                            top_evecs = top_evecs, device = DEVICE)
                        wandb.log({'sharpness' : sharpness})
                        print(f'shaprness : {sharpness}')
                    SHARPNESS_TIMER += 1

                # Backwards pass and optimization
                # TODO: get rid of the if clause
                if algo == 'SGD':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                elif algo == 'Bulk-SGD':
                    loss.backward()
                    dataset = TensorDataset(images, labels)
                    optimizer.calculate_evecs(model, criterion, dataset)
                    optimizer.step()
                    optimizer.zero_grad()
                elif algo == 'Top-SGD':
                    loss.backward()
                    dataset = TensorDataset(images, labels)
                    optimizer.calculate_evecs(model, criterion, dataset)
                    optimizer.step()
                    optimizer.zero_grad()
                elif algo == 'prev_Bulk-SGD':
                    loss.backward()
                    dataset = TensorDataset(images, labels)
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    raise NotImplementedError()
                
                if args.save_evecs and hasattr(optimizer, 'evecs') and optimizer.evecs is not None :
                    if TOP_EVEC_TIMER % TOP_EVEC_RECORD_FREQ == 0:
                        TOP_EVECS.append((cur_training_steps, phase, optimizer.evecs.clone()))
                    TOP_EVEC_TIMER += 1

                if 'holdouts' in kwargs:
                    if holdout_acc_timer % holdout_acc_freq == 0:
                        for i, holdout_dataset in enumerate(kwargs['holdouts']):
                                holdout_loader = DataLoader(dataset=holdout_dataset, 
                                                            batch_size=args.batch_size,
                                                            shuffle=True, pin_memory=True)
                                holdout_accuracy = eval(holdout_loader, model, verbose = False)
                                wandb.log({f"exp_{i}_accuracy" : holdout_accuracy})
                    holdout_acc_timer += 1
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {(loss_sum / batches):.4f}')
        wandb.log({'loss': (loss_sum / batches)})
        per_epoch_losses.append(loss_sum / batches)

    return losses, per_epoch_losses, training_steps_per_batch, training_steps_per_epoch

def eval(test_loader, model, verbose = True):
    model.eval()  
    with torch.no_grad():  
        correct = 0
        total = 0
        for batch in test_loader:
            images, labels, *rest = batch

            images = images.to(DEVICE) 
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        if verbose:
            print(f'Accuracy on the test set: {100 * correct / total:.2f}%')
            wandb.log({'Accuracy' : 100 * correct / total})
    return 100 * correct / total

def projected_training(args):
    # Hyperparameters
    batch_size = args.batch_size
    lr = args.lr

    # Load the dataset
    if args.dataset == 'MNIST_5k':
        input_size = 28 * 28
        output_size = 10
        num_classes = 10
        transform = transforms.Compose([transforms.ToTensor(),]) 
                                        #transforms.Normalize((0.,), (1.,))])  # is this normalization good?
        train_dataset = datasets.MNIST(root='./data', 
                                            train=True, 
                                            transform=transform, 
                                            download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        
        train_size = 5000 if not args.debug else 2 * batch_size
        test_size = len(test_dataset) if not args.debug else 2 * batch_size
        
        train_indices = np.random.choice(len(train_dataset), train_size, replace=False)
        train_dataset = Subset(train_dataset, train_indices)
        test_indices = np.random.choice(len(test_dataset), test_size, replace=False)
        test_dataset = Subset(test_dataset, test_indices)
        
        print(f'Train dataset with {train_size} samples')
        print(f'Test dataset with {test_size} samples')
    elif args.dataset == 'MNIST_full':
        input_size = 28 * 28
        output_size = 10
        num_classes = 10
        transform = transforms.Compose([transforms.ToTensor(),]) 
                                        #transforms.Normalize((0.,), (1.,))])  # is this normalization good?
        train_dataset = datasets.MNIST(root='./data', 
                                            train=True, 
                                            transform=transform, 
                                            download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        
        train_size = len(train_dataset)
        test_size = len(test_dataset) if not args.debug else 2 * batch_size
        
        print(f'Train dataset with {train_size} samples')
        print(f'Test dataset with {test_size} samples')
    else:   
        raise NotImplementedError()    

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Initialize the model, loss_function, and optimizer
    model = get_model(input_size, output_size, args, DEVICE).to(DEVICE)
    wandb.watch(model, log_graph=True)
    if args.loss == 'cross_entropy_loss':
        criterion = nn.CrossEntropyLoss()   
    elif args.loss == 'MSE':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError()
    optimizer = get_optimizer(args, model)

    summary(model, input_size=(28, 28),device=DEVICE)

    # Warm-up loop
    print('Started warm-up training...')
    warm_up_losses = train(train_loader, model, 
                           criterion, optimizer, 
                           lr, num_epochs=args.warm_up_epochs, algo='SGD', 
                           num_classes=num_classes, 
                           phase="warm-up", 
                           args=args)
    warm_up_training_steps = 0 if len(warm_up_losses[2]) == 0 else warm_up_losses[2][-1]
    print('Finished warm-up training!')
    
    print('Post warm-up evalution...')
    warm_up_accuracy = eval(test_loader, model)   
    if hasattr(optimizer, '_warm_up'):
        optimizer._warm_up = False
    # Training loop 
    print('Started training...')
    train_losses = train(train_loader, model, 
                         criterion, optimizer, lr, 
                         num_epochs=args.epochs,algo=args.algo, 
                         cur_training_steps=warm_up_training_steps, 
                         phase="training", args = args)
    print('Finished training!')

    print('Final evaluation')
    final_accuracy = eval(test_loader, model)

    save_results(args, warm_up_losses, train_losses, warm_up_accuracy, final_accuracy, warm_up_training_steps, TOP_EVECS)

def cl_task(args):
    """each experience: warm-up-epochs x SGD + epochs x algo
    -------
    Run with the following command: 
     ```python train.py --task cl_task \
        --epochs 1  \
        --hidden_sizes 100 100 100 \
        --activation 'relu' \
        --warm_up_epochs 0   \
        --algo prev_Bulk-SGD   \
        --plot_losses  \
        --lr 0.01 \
        --seed 125 \
        --loss MSE \
        --n_experiences 3 \
        --n_bulk_batches 1 \
    ```
    -------
    Very slow!! -> Change the train_subset_size

    """

    # Hyperparameters
    batch_size = args.batch_size
    lr = args.lr
    train_subset_size = None
    holdout_size = 500

    # Load the dataset
    benchmark = PermutedMNIST(n_experiences=args.n_experiences)
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream
    input_size = 28 * 28
    output_size = 10

    holdout_datasets = []
    for experience in train_stream:
        print("Start of task ", experience.task_label)
        print('Classes in this task:', experience.classes_in_this_experience)

        current_training_set = experience.dataset
        print('Task {}'.format(experience.task_label))
        print('This task contains', train_subset_size, 'training examples')

        current_test_set = test_stream[experience.current_experience].dataset
        print('This task contains', len(current_test_set), 'test examples')

        holdout_datasets.append(get_holdout_dataset(current_test_set, holdout_size))

    # Define CustomCLStrategy
    class CustomCLStrategy(SupervisedTemplate):
        """Mostly copied from https://avalanche-api.continualai.org/en/v0.5.0/_modules/avalanche/training/supervised/strategy_wrappers.html#Naive"""

        def __init__(
            self,
            *,
            model,
            optimizer,
            criterion,
            lr,
            train_mb_size,
            train_epochs,
            eval_mb_size,
            device,
            **base_kwargs
        ):
            super().__init__(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_mb_size=train_mb_size,
                train_epochs=train_epochs,
                eval_mb_size=eval_mb_size,
                device=device,
                **base_kwargs
            )
            self.lr = lr
            self._criterion = criterion

        def train(self, experience, exp_id=0, prev_evals=[], prev_evecs=[], holdouts=[]):

            train_dataloader = get_partial_dataloader(
                experience.dataset, train_subset_size, self.train_mb_size)
            warm_up_epochs = args.warm_up_epochs
            train_epochs = args.epochs

            # TODO: change to a single list
            print('Started warm-up training...')
            warm_up_losses = train(train_dataloader, self.model, self._criterion,
                                   self.optimizer, self.lr, warm_up_epochs, 'SGD', 0, phase=f'{exp_id}', args=args, holdouts=holdouts)
            warm_up_training_steps = 0 if len(
                warm_up_losses[2]) == 0 else warm_up_losses[2][-1]
            print('Started training...')
            if exp_id == 0:
                train_losses = train(train_dataloader, self.model, self._criterion, self.optimizer,
                                     self.lr, train_epochs, 'SGD', cur_training_steps=warm_up_training_steps, phase=f'{exp_id}', args=args, holdouts=holdouts)
            else:
                if hasattr(optimizer, '_warm_up'):
                    optimizer._warm_up = False
                train_losses = train(train_dataloader, self.model, self._criterion, self.optimizer, self.lr, train_epochs,
                                     args.algo, cur_training_steps=warm_up_training_steps, evals=prev_evals, evecs=prev_evecs, phase=f'{exp_id}', args=args, holdouts=holdouts)
                if hasattr(optimizer, '_warm_up'):
                    optimizer._warm_up = True

            return warm_up_losses, train_losses, warm_up_training_steps

    # Initialize the model, loss_function, and optimizer, strategy
    if args.algo == "gpm":
        bias = False
        print("Important: GPM does not support bias")
    else:
        bias = True

    model = get_model(input_size, output_size, args, DEVICE, bias).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, model)
    if args.algo in baselines:
        hyperparams = {
            "ewc" : {
                "ewc_lambda": 10.0,
                "train_mb_size": batch_size,
                "train_epochs": args.epochs,
                "eval_mb_size": batch_size,
            },
            "agem" : {
                "patterns_per_exp": 500,
                "train_mb_size": batch_size,
                "train_epochs": args.epochs,
                "eval_mb_size": batch_size,
            },
            "gpm" : {
                "train_mb_size": batch_size,
                "train_epochs": args.epochs,
                "eval_mb_size": batch_size,
                "lr": args.lr,
            }
        }

        final_accuracy, all_train_losses = run_avalanche(args, args.algo, hyperparams[args.algo], model, optimizer, criterion, benchmark, holdout_datasets)
        all_warm_up_losses = [([], [], [], [])]*len(all_train_losses)
        save_results_cl(args, all_warm_up_losses, all_train_losses, final_accuracy, top_evecs=TOP_EVECS)
        return
    
    eval_plugin = EvaluationPlugin(
        StreamAccuracy(),
    )
    custom_cl_strategy = CustomCLStrategy(
        model, optimizer, criterion, lr, train_mb_size=batch_size, train_epochs=5, eval_mb_size=batch_size, evaluator = eval_plugin, device=DEVICE)

    # for the overlap experiments 
    if args.log_overlaps and hasattr(optimizer, '_log_overlaps'):
        optimizer._log_overlaps = True

    # Training loop
    all_warm_up_losses = []
    all_train_losses = []
    all_warm_up_training_steps = []
    experience_ids = []
    cumulative_steps = 0  # Tracks total training steps across experiences
    prev_evals = None
    prev_evecs = None

    cur_holdouts = []
    for exp_id, experience in enumerate(train_stream):
        cur_holdouts.append(holdout_datasets[exp_id])

        print(f"Start of experience {exp_id + 1}: {experience}")

        # Train on current experience
        warm_up_losses, train_losses, warm_up_training_steps = custom_cl_strategy.train(
            experience, exp_id, prev_evals, prev_evecs, cur_holdouts)
        print("Training completed.")

        all_warm_up_losses.append(warm_up_losses)
        all_train_losses.append(train_losses)
        all_warm_up_training_steps.append(warm_up_training_steps)

        if hasattr(optimizer, 'append_evecs'):
            subset_indices = np.random.choice(len(experience.dataset), args.hessian_subset_size, replace=False)
            partial_dataset = Subset(experience.dataset, subset_indices)
    
            optimizer.append_evecs(model, criterion, partial_dataset)

    print('Computing accuracy on the test set')
    final_accuracy = custom_cl_strategy.eval(test_stream)

    print(f'Average accuracy: {final_accuracy["Top1_Acc_Stream/eval_phase/test_stream/Task000"] * 100.}')
    wandb.log({'ACC' : final_accuracy["Top1_Acc_Stream/eval_phase/test_stream/Task000"] * 100.})

    save_results_cl(args, all_warm_up_losses, all_train_losses, final_accuracy, top_evecs=TOP_EVECS)
    print(final_accuracy)

def main(args):
    if not args.debug:
        wandb_mode = "online"
    else:
        wandb_mode = "disabled"
    wandb.init(
        project="ETH-DL-Project",
        config=args,
        mode=wandb_mode
    )
    print(f'DEVICE = {DEVICE}')

    if args.task == 'projected_training':
        projected_training(args)
    elif args.task == 'cl_task':
        cl_task(args)
    else:
        raise NotImplementedError()        

    wandb.finish()

if __name__ == "__main__": 
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    print(f"device: {DEVICE}")

    args = arg_parser()
    seed_everything(args.seed)

    main(args) 