# https://github.com/sahagobinda/GPM/blob/main/main_pmnist.py

import torch
from torch.utils.data import DataLoader, Subset

import numpy as np
from avalanche.training.templates import SupervisedTemplate

def get_representation_matrix (net, device, x, y=None): 
    # Collect activations by forward pass
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:300] # Take random training samples
    example_data = x[b].view(-1,28*28)
    example_data = example_data.to(device)
    example_out  = net(example_data)
    
    batch_list=[300] * net.n_lin #batch_list=[300,300,300] 
    mat_list=[] # list contains representation matrix of each layer
    act_key=list(net.act.keys())

    for i in range(len(act_key)):
        bsz=batch_list[i]
        act = net.act[act_key[i]].detach().cpu().numpy()
        activation = act[0:bsz].transpose()
        mat_list.append(activation)

    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_list)):
        print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
    print('-'*30)
    return mat_list    


def update_GPM (model, mat_list, threshold, feature_list=[],):
    print ('Threshold: ', threshold) 
    if not feature_list:
        # After First Task 
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
            feature_list.append(U[:,0:r])
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            # Projected Representation (Eq-8)
            act_hat = activation - np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
            U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total               
            accumulated_sval = (sval_total-sval_hat)/sval_total
            
            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                continue
            # update GPM
            Ui=np.hstack((feature_list[i],U[:,0:r]))  
            if Ui.shape[1] > Ui.shape[0] :
                feature_list[i]=Ui[:,0:Ui.shape[0]]
            else:
                feature_list[i]=Ui
    
    print('-'*40)
    print('Gradient Constraints Summary')
    print('-'*40)
    for i in range(len(feature_list)):
        print ('Layer {} : {}/{}'.format(i+1,feature_list[i].shape[1], feature_list[i].shape[0]))
    print('-'*40)
    return feature_list  

def get_dataset(loader):
    # Assuming you have a DataLoader named 'dataloader'
    data_list = []
    labels_list = []

    for data, labels, *rest in loader:
        data_list.append(data)
        labels_list.append(labels)

    # Concatenate all batches into a single tensor
    all_data = torch.cat(data_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    return all_data, all_labels


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

# Define CustomCLStrategy
class GPM(SupervisedTemplate):
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
        **base_kwargs
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            **base_kwargs
        )
        self.lr = lr
        self._criterion = criterion

    def epoch_loss(self, losses):        
        # Compute average loss for the epoch
        sum_losses = sum(losses)
        num_minibatches = len(losses)
        epoch_loss = sum_losses / num_minibatches
        return epoch_loss

    def train_epoch(self, model, device, x, y, optimizer,criterion):
        train_mb_size = self.train_mb_size
        model.train()
        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).to(device)

        losses = []
        training_steps = []
        n = 0
        # Loop batches
        for i in range(0,len(r),train_mb_size):
            if i+train_mb_size<=len(r): b=r[i:i+train_mb_size]
            else: b=r[i:]
            data = x[b].view(-1,28*28)
            data, target = data.to(device), y[b].to(device)
            optimizer.zero_grad()        
            output = model(data)
            loss = criterion(output, target)      
            losses.append(loss.item())  
            loss.backward()
            optimizer.step()
            n += len(b)
            training_steps.append(n)
        return losses, training_steps

    def train_epoch_projected(self, model,device,x,y,optimizer,criterion,feature_mat):
        train_mb_size = self.train_mb_size
        model.train()
        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).to(device)

        losses = []
        training_steps = []
        n = 0
        # Loop batches
        for i in range(0,len(r),train_mb_size):
            if i+train_mb_size<=len(r): b=r[i:i+train_mb_size]
            else: b=r[i:]
            data = x[b].view(-1,28*28)
            data, target = data.to(device), y[b].to(device)
            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, target)
            losses.append(loss.item())  

            loss.backward()
            # Gradient Projections 
            for k, (m,params) in enumerate(model.named_parameters()):
                sz =  params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                        feature_mat[k]).view(params.size())
            optimizer.step()
            n += len(b)
            training_steps.append(n)
        return losses, training_steps
        
    def train(self, experience):
        model = self.model
        criterion = self._criterion
        device = self.device
        optimizer = self.optimizer
        task_id = experience.current_experience # experience id
        lr = self.lr

        train_subset_size = None

        train_dataloader = get_partial_dataloader(
                    experience.dataset, train_subset_size, self.train_mb_size)
        # specify threshold hyperparameter
        #threshold = np.array([0.95,0.99,0.99]) 
        threshold = np.array([0.95]*model.n_lin)
    
        xtrain, ytrain = get_dataset(train_dataloader)

        total_losses = []
        total_epoch_losses = []

        training_steps_per_batch = []
        training_steps_per_epoch = []

        self.curr_steps = 0

        #lr = args.lr 
        if task_id==0:
            #print ('Model parameters ---')
            #for k_t, (m, param) in enumerate(model.named_parameters()):
            #    print (k_t,m,param.shape)
            #print ('-'*40)

            self.feature_list =[]
            for epoch in range(1, self.train_epochs+1):
                # Train
                losses, training_steps = self.train_epoch(model, device, xtrain, ytrain, optimizer, criterion)
                training_steps_per_batch += [x + self.curr_steps for x in training_steps]
                total_losses += losses
            total_epoch_losses.append(self.epoch_loss(losses))
            self.curr_steps = training_steps_per_batch[-1]
            training_steps_per_epoch.append(self.curr_steps)

            # Memory Update  
            mat_list = get_representation_matrix (model, device, xtrain, ytrain)
            self.feature_list = update_GPM (model, mat_list, threshold, self.feature_list)
            

        else:
            feature_mat = []
            # Projection Matrix Precomputation
            for i in range(len(model.act)):
                Uf=torch.Tensor(np.dot(self.feature_list[i],self.feature_list[i].transpose())).to(device)
                print('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
                feature_mat.append(Uf)
            print ('-'*40)
            for epoch in range(1,self.train_epochs+1):
                # Train 
                losses, training_steps = self.train_epoch_projected(model,device,xtrain, ytrain,optimizer,criterion,feature_mat)
                training_steps_per_batch += [x + self.curr_steps for x in training_steps]
                total_losses += losses
            total_epoch_losses.append(self.epoch_loss(losses))
            self.curr_steps = training_steps_per_batch[-1]
            training_steps_per_epoch.append(self.curr_steps)
            # Memory Update 
            mat_list = get_representation_matrix (model, device, xtrain, ytrain)
            self.feature_list = update_GPM (model, mat_list, threshold, self.feature_list)

        return total_losses, total_epoch_losses, training_steps_per_batch, training_steps_per_epoch