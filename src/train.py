import os 
import pathlib
import pickle
from datetime import datetime
from argparse import ArgumentParser, BooleanOptionalAction

from utilities import get_hessian_eigenvalues, timeit, time_block
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt


DEVICE = 'cpu'

def arg_parser():
    parser = ArgumentParser(description='Train')

    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--debug', action=BooleanOptionalAction, default=False)
    parser.add_argument('--storage', type=pathlib.Path, default='../storage')
    parser.add_argument('--save_results', action=BooleanOptionalAction, default=True)


    parser.add_argument('--epochs', type=int, default=5, required=True)
    parser.add_argument('--dataset', choices = ['MNIST_5k'], default='MNIST_5k')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)

    # Model parameters 
    parser.add_argument('--model', choices = ['MLP'], default='MLP')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', required=False)
    parser.add_argument('--num_hidden_layers', type=int, nargs='+', required=False)
    parser.add_argument('--activation', choices = ['relu','tanh'], default='relu')

    # projected_training args
    parser.add_argument('--warm_up_epochs', type=int, default=0, required=False)
    parser.add_argument('--algo', choices=['SGD', 'Bulk-SGD', 'Top-SGD'], default='SGD')
    parser.add_argument('--plot_losses', action=BooleanOptionalAction, default=False)

    args = parser.parse_args()
    return args

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
                    layers.append(nn.ReLU())
                    in_size = hidden_size
                
                layers.append(nn.Linear(in_size, output_size).to(DEVICE))
                self.model = nn.Sequential(*layers)
                
            def forward(self, x):
                x = x.view(-1, 28 * 28)
                return self.model(x)
            
        return MLP(input_size, output_size, args.hidden_sizes)
    else:
        raise NotImplementedError()
    
def projected_step_bulk(model, loss, criterion, dataset, batch_size, lr):
    evals, evecs = get_hessian_eigenvalues(model, criterion, dataset, 
                                           physical_batch_size=batch_size, 
                                           neigs=10, device=DEVICE)
    evecs.transpose_(1, 0)
    
    with torch.no_grad(): 
        grad = torch.autograd.grad(loss, inputs=model.parameters(), create_graph=True)
        vec_grad = parameters_to_vector(grad)
        step = vec_grad.detach() 
        for vec in evecs:
            step -= torch.dot(vec_grad, vec) * vec
        vec_params = parameters_to_vector(model.parameters())
        vec_params -= step * lr
        vector_to_parameters(vec_params, model.parameters())
        model.zero_grad()

def projected_step_top(model, loss, criterion, dataset, batch_size, lr):
    evals, evecs = get_hessian_eigenvalues(model, criterion, dataset,
                                           physical_batch_size=batch_size,
                                           neigs=10, device=DEVICE)
    evecs.to(DEVICE).transpose_(1, 0)
    
    with torch.no_grad():  # Ensure we don’t track these operations for gradient computation
        grad = torch.autograd.grad(loss, inputs=model.parameters(), create_graph=True)
        vec_grad = parameters_to_vector(grad).to(DEVICE)
        step = torch.Tensor(vec_grad.shape).to(DEVICE)
        for vec in evecs:
            step += torch.dot(vec_grad, vec) * vec
        vec_params = parameters_to_vector(model.parameters())
        vec_params -= step * lr
        vector_to_parameters(vec_params, model.parameters())
        model.zero_grad()

def train(train_loader, model, criterion, optimizer, lr, num_epochs : int, algo : str = 'SGD'):
    losses = []
    per_epoch_losses = []
    num_images = 0
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        loss_sum = 0.0
        batches = 0.0
        with time_block("Training epoch"):
            for images, labels in train_loader:
                images = images.to(DEVICE) 
                labels = labels.to(DEVICE)
                k = len(images)

                # Forward pass 
                outputs = model(images).to(DEVICE)
                loss = criterion(outputs, labels)
                loss_sum += loss.item()
                batches += 1
                losses.append(loss.item() / k)
                num_images += k

                # Backwards pass and optimization
                if algo == 'SGD':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                elif algo == 'Bulk-SGD':
                    dataset = TensorDataset(images, labels)
                    projected_step_bulk(model, loss, criterion, dataset, batch_size=64, lr=lr)
                elif algo == 'Top-SGD':
                    dataset = TensorDataset(images, labels)
                    projected_step_top(model, loss, criterion, dataset, batch_size=64, lr=lr)
                else:
                    raise NotImplementedError()
                
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {(loss_sum / num_images):.4f}')
        per_epoch_losses.append(loss_sum / num_images)

    return losses, per_epoch_losses

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
    return 100 * correct / total

def projected_training(args):
    # Hyperparameters
    batch_size = args.batch_size
    lr = args.lr

    # Load the dataset
    if args.dataset == 'MNIST_5k':
        input_size = 28 * 28
        output_size = 10
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
    else:   
        raise NotImplementedError()    

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Initialize the model, loss_function, and optimizer
    model = get_model(input_size, output_size, args).to(DEVICE)
    criterion = nn.CrossEntropyLoss()   
    optimizer = optim.SGD(model.parameters(), lr=lr)

    summary(model, input_size=(28, 28),device=DEVICE)

    # Warm-up loop
    print('Started warm-up training...')
    warm_up_losses = train(train_loader, model, 
                           criterion, optimizer, 
                           lr, num_epochs=args.warm_up_epochs, algo='SGD')
    print('Finished warm-up training!')
    
    print('Post warm-up evalution...')
    warm_up_accuracy = eval(test_loader, model)   

    # Training loop 
    print('Started training...')
    train_losses = train(train_loader, model, 
                         criterion, optimizer, lr, 
                         num_epochs=args.epochs, algo=args.algo)
    print('Finished training!')

    print('Final evaluation')
    final_accuracy = eval(test_loader, model)

    per_batch_losses = warm_up_losses[0] + train_losses[0]
    per_epoch_losses = warm_up_losses[1] + train_losses[1]
    output_name = f"{args.dataset}_{args.model}_{args.activation}_{args.algo}"
    if args.hidden_sizes:
        output_name += "_hidden_sizes_" + "-".join(map(str, args.hidden_sizes))
    if args.num_hidden_layers:
        output_name += "_num_hidden_layers_" + "-".join(map(str, args.num_hidden_layers))

    if args.plot_losses:
        plt.figure(1)
        plt.axvline(x=len(warm_up_losses[0]), color='red', linestyle='--', label="End of warm-up")
        plt.plot(np.log(per_batch_losses))
        plt.title('Per Batch Losses (log-scale)')
        plt.legend()

        plt.figure(2)
        plt.axvline(x=len(warm_up_losses[1]), color='red', linestyle='--', label="End of warm-up")
        plt.plot(np.log(per_epoch_losses))
        plt.title('Per Epoch Losses (log-scale)')
        plt.legend()

        plt.show()
        output_dir = "../plots"

        plot_name = output_name+ '.png'
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, plot_name)
        plt.savefig(save_path)


    results = {}
    results['args'] = args
    results['warm_up_losses_batch'] = warm_up_losses[0]
    results['warm_up_losses_epoch'] = warm_up_losses[1]
    results['train_losses_batch'] = train_losses[0]
    results['train_losses_epoch'] = train_losses[1]
    results['warm_up_accurary'] = warm_up_accuracy
    results['final_accuracy'] = final_accuracy

    if args.save_results and not args.debug:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f'{args.storage}/projected_training_{current_time}_{output_name}.pkl'
        with open(file_name, "wb") as file:
            pickle.dump(results, file)
        print(f'Results saved in file_name!')

def main(args):
    print(f'DEVICE = {DEVICE}')

    if args.task == 'projected_training':
        projected_training(args)
    else:
        raise NotImplementedError()        


if __name__ == "__main__": 
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    print(f"device: {DEVICE}")

    args = arg_parser()

    main(args) 