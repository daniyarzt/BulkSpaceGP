from ast import parse
import os 
import pathlib
import random
from argparse import ArgumentParser, BooleanOptionalAction

from utilities import get_hessian_eigenvalues, timeit, time_block, save_results, save_results_cl
from proj_optimizers import BulkSGD, TopSGD, CLBulkSGD
from torch.nn.utils import parameters_to_vector, vector_to_parameters

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

DEVICE = 'cpu'

TOP_EVECS = []

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
    parser.add_argument('--algo', choices=['SGD', 'Bulk-SGD', 'Top-SGD', 'prev_Bulk-SGD'], default='SGD')
    parser.add_argument('--plot_losses', action=BooleanOptionalAction, default=False)

    # CL task 
    parser.add_argument('--n_experiences', type=int, default=2, required=False)

    # Overlap args 
    parser.add_argument('--save_evecs', action=BooleanOptionalAction, default=False)

    args = parser.parse_args()
    return args

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(input_size : int, output_size : int, args) -> nn.Module:
    # Model definitions could be moved to a separate file...  
    if args.model == 'MLP':
        # Define the MLP model 
        class MLP(nn.Module):
            def __init__(self, input_size, output_size, hidden_sizes):
                super(MLP, self).__init__()
                
                layers = []
                in_size = input_size 

                for hidden_size in hidden_sizes:
                    layers.append(nn.Linear(in_size, hidden_size))
                    if args.activation == 'relu':
                        layers.append(nn.ReLU())
                    elif args.activation == 'tanh':
                        layers.append(nn.Tanh())
                    in_size = hidden_size
                
                layers.append(nn.Linear(in_size, output_size).to(DEVICE))
                self.model = nn.Sequential(*layers)
                
            def forward(self, x):
                x = x.view(-1, 28 * 28)
                return self.model(x)
            
        return MLP(input_size, output_size, args.hidden_sizes)
    else:
        raise NotImplementedError()
    
def get_optimizer(args, model):
    lr = args.lr
    batch_size = args.batch_size

    if args.algo == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif args.algo == "Bulk-SGD":
        optimizer = BulkSGD(model.parameters(), lr=lr, batch_size=batch_size, device=DEVICE)
    elif args.algo == "Top-SGD":
        optimizer = TopSGD(model.parameters(), lr=lr, batch_size=batch_size, device=DEVICE)
    elif args.algo == "prev_Bulk-SGD":
        optimizer = CLBulkSGD(model.parameters(), lr=lr, batch_size=batch_size, device=DEVICE)
    else:
        raise NotImplementedError()
    return optimizer

def get_partial_dataloader(dataset, subset_size, batch_size):
    subset_indices = np.random.choice(len(dataset), subset_size, replace=False)
    partial_dataset = Subset(dataset, subset_indices)
    return DataLoader(dataset=partial_dataset, batch_size=batch_size,
                                    shuffle=True, pin_memory=True)

def train(train_loader, model, criterion, optimizer, lr, num_epochs: int, algo: str = 'SGD', cur_training_steps=0, num_classes=10, evals=[], evecs=[], phase = "", args = None):
    save_evecs = True
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
            for batch in train_loader:
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
                    dataset = TensorDataset(images, labels)
                    _, cur_evecs = get_hessian_eigenvalues(model, criterion, dataset,
                                                   physical_batch_size=len(images),
                                                   neigs=10, device=DEVICE)
                    cur_evecs.transpose_(1, 0)
                    TOP_EVECS.append((cur_training_steps, phase, cur_evecs.clone()))

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
                if args.save_evecs:
                    if hasattr(optimizer, 'evecs') and optimizer.evecs is not None:
                        TOP_EVECS.append((cur_training_steps, phase, optimizer.evecs.clone()))
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {(loss_sum / batches):.4f}')
        wandb.log({'loss': (loss_sum / batches)})
        per_epoch_losses.append(loss_sum / batches)

    return losses, per_epoch_losses, training_steps_per_batch, training_steps_per_epoch

def eval(test_loader, model):
    model.eval()  
    with torch.no_grad():  
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(DEVICE) 
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
    model = get_model(input_size, output_size, args).to(DEVICE)
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
     python train.py --task cl_task   --epochs 4     --hidden_sizes 200 200 200 --activation 'tanh'     --warm_up_epochs 1     --algo prev_Bulk-SGD     --plot_losses     --lr 0.01     --seed 125     --loss MSE
    -------
    Very slow!! -> Change the train_subset_size

    """

    # Hyperparameters
    batch_size = args.batch_size
    lr = args.lr
    train_subset_size = 5000

    # Load the dataset
    benchmark = PermutedMNIST(n_experiences=args.n_experiences)
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream
    input_size = 28 * 28
    output_size = 10

    for experience in train_stream:
        print("Start of task ", experience.task_label)
        print('Classes in this task:', experience.classes_in_this_experience)

        current_training_set = experience.dataset
        print('Task {}'.format(experience.task_label))
        print('This task contains', train_subset_size, 'training examples')

        current_test_set = test_stream[experience.current_experience].dataset
        print('This task contains', len(current_test_set), 'test examples')

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

        def train(self, experience, exp_id=0, prev_evals=[], prev_evecs=[]):

            train_dataloader = get_partial_dataloader(experience.dataset, train_subset_size, self.train_mb_size)
            warm_up_epochs = args.warm_up_epochs
            train_epochs = args.epochs

            # TODO: change to a single list 
            print('Started warm-up training...')
            warm_up_losses = train(train_dataloader, self.model, self._criterion,
                                   self.optimizer, self.lr, warm_up_epochs, 'SGD', 0)
            warm_up_training_steps = 0 if len(
                warm_up_losses[2]) == 0 else warm_up_losses[2][-1]
            print('Started training...')
            if exp_id == 0:
                train_losses = train(train_dataloader, self.model, self._criterion, self.optimizer,
                                     self.lr, train_epochs, 'SGD', cur_training_steps=warm_up_training_steps)
            else:
                if hasattr(optimizer, '_warm_up'):
                    optimizer._warm_up = False
                train_losses = train(train_dataloader, self.model, self._criterion, self.optimizer, self.lr, train_epochs,
                                     args.algo, cur_training_steps=warm_up_training_steps, evals=prev_evals, evecs=prev_evecs)
                if hasattr(optimizer, '_warm_up'):
                    optimizer._warm_up = True

            return warm_up_losses, train_losses, warm_up_training_steps

    # Initialize the model, loss_function, and optimizer, strategy
    model = get_model(input_size, output_size, args).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, model)
    custom_cl_strategy = CustomCLStrategy(
        model, optimizer, criterion, lr, train_mb_size=batch_size, train_epochs=5, eval_mb_size=batch_size, device=DEVICE, )

    # Training loop
    all_warm_up_losses = []
    all_train_losses = []
    all_warm_up_training_steps = []
    experience_ids = []
    cumulative_steps = 0  # Tracks total training steps across experiences
    prev_evals = None
    prev_evecs = None

    for exp_id, experience in enumerate(train_stream):

        print(f"Start of experience {exp_id + 1}: {experience}")

        # Train on current experience
        warm_up_losses, train_losses, warm_up_training_steps = custom_cl_strategy.train(
            experience, exp_id, prev_evals, prev_evecs)
        print("Training completed.")

        all_warm_up_losses.append(warm_up_losses)
        all_train_losses.append(train_losses)
        all_warm_up_training_steps.append(warm_up_training_steps)

        if hasattr(optimizer, 'append_evecs'):
            train_dataloader = get_partial_dataloader(experience.dataset, train_subset_size, args.batch_size)
            last_batch = list(train_dataloader)[-1]
            images, labels, *rest = last_batch
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            dataset = TensorDataset(images, labels)

            optimizer.append_evecs(model, criterion, dataset)

    print('Computing accuracy on the test set')
    final_accuracy = custom_cl_strategy.eval(test_stream)
    save_results_cl(args, all_warm_up_losses, all_train_losses, final_accuracy)

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
    seed_everything()

    main(args) 