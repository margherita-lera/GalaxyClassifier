# %%
import optuna
#from tqdm import tqdm
import logging
import sys
import torch
import os
from datetime import datetime
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torch.utils.data
#from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import random_split, Dataset, DataLoader
from PIL import Image
import zookeeper as zk  # convool_size & mappy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

# %%
input_size = 128
for convool_layer in range(5):
    conv_outsize = zk.convool_size(input_size, kernel_size=3, stride=1)
    pool_outsize = zk.convool_size(conv_outsize, kernel_size=2, stride=2)
    input_size = pool_outsize

pool_outsize

# %% [markdown]
# ### Define Dataset function

# %%
class GalaxyJungle(Dataset):
    
    #the init function initializes the directory containing the image,
    #the annotations file,
    #and both transforms
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, is_rgb=False):
        self.rgb = is_rgb
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    #returns number of samples in the dataset
    def __len__(self):
        return (self.img_labels).shape[0]

    #loads a sample from the dataset
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0])) + '.jpg'
        #retrieves the image
        image = Image.open(img_path)
        if not self.rgb: image = image.convert('L')
        #retrieves corresponding label
        label = self.img_labels.iloc[idx, 1:]
        #if possible, transform the image and the label into a tensor.
        if self.transform:
            image = self.transform(image).type(torch.float32)
        label = torch.tensor(label.values, dtype=torch.float32)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, self.img_labels.iloc[idx, 0]
    

transfs = transforms.Compose([
    transforms.ToTensor(), # Riscala le immagini tra 0 e 1
    transforms.CenterCrop(312),
    transforms.Resize(256),
    # sarebbe interessante implementare un random crop prima del center crop per decentrare un poco le immagini????
    #transforms.RandomHorizontalFlip(), # horizontal flip
    #transforms.RandomVerticalFlip(), # vertical flip
              #CROP
    ]) #transforms.compose per fare una pipe di transformazioni

# %% [markdown]
# ## NEURAL NETWORK

# %%
class GalaxyNet(nn.Module):
    def __init__(self, n_conv_layers, num_filters, num_neurons, activation, is_rgb=False, verbose=False):
        super().__init__()
        rgb = 3 if is_rgb else 1
        input_size = 256
        num_labels = 37
        self.loss_dict = {'batch' : [], 'epoch' : [], 'vbatch' : [], 'vepoch' : []}
        self.activation = activation
        
        self.num_convs = n_conv_layers


        stride = 2
        kernel_size = 3
        kernel_size_pool = 2
        
        ## convolutional layers
        self.convs = nn.Sequential(
            nn.Conv2d(rgb, num_filters[0], kernel_size=kernel_size, stride=stride, bias=False),
            self.activation(),
            nn.BatchNorm2d(num_filters[0]),
            nn.MaxPool2d(kernel_size=kernel_size_pool)
            )
        output_size = (input_size - kernel_size + stride) // (stride*kernel_size_pool)

        if verbose: print('output size after first conv layer: ', output_size)

        for i in range(1,n_conv_layers):
            self.convs.append(nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=kernel_size, stride=stride, bias=False))
            self.convs.append(self.activation())
            self.convs.append(nn.BatchNorm2d(num_filters[i]))
            self.convs.append(nn.MaxPool2d(kernel_size=kernel_size_pool))

            output_size = (output_size - kernel_size + stride) // (stride*kernel_size_pool) #padding 0, dilation = 1
            if verbose: 
                if i != n_conv_layers - 1: print('output size after conv layer {}: '.format(i), output_size)
                else: 
                    cane = 2#print('output size of the last conv layer: ', output_size) #print('len self convs: ',len(self.convs))
        if output_size in (0, -1): output_size = 1
        #print(output_size)
        
        #self.convs.append(nn.dropout(p= value)) ## to be added in the future to test claims of BatchnOrm paper
        
        self.out_feature = num_filters[-1] *output_size * output_size # output size of the last conv layer, should be 38
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.out_feature, num_neurons),
            self.activation(),
            nn.Linear(num_neurons, num_labels)
            )
        
        self.init_weights()


    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x)
        # x = nn.Sigmoid()(x)
        return x


        

    def init_weights(self):
        if self.activation == nn.ReLU: # perchè kaiming normal e non uniform??1
            nonlin = 'relu'
            a = 0
        elif self.activation == nn.LeakyReLU:
            nonlin = 'leaky_relu'
            a = .01
        #print(a)
        for i in range(0, self.num_convs*4, 4): nn.init.kaiming_normal_(self.convs[i].weight, a=a, nonlinearity=nonlin)
        for i in (1, -1): nn.init.constant_(self.fc[i].bias, 0)
        nn.init.kaiming_normal_(self.fc[1].weight, a=a, nonlinearity=nonlin)
        nn.init.xavier_uniform_(self.fc[-1].weight)
        return #print('weights initialized with {}'.format(self.activation))         
        

    

    
    def log_the_loss(self, item,epoch=False): # per avere una history della loss???
        verbose=False
        train = self.__getstate__()['training']
        if verbose: print(train)
        if epoch and train:
            self.loss_dict['epoch'].append(item) ### get state of the model so you can ditch the validation parameter
        elif not epoch and train:
            self.loss_dict['batch'].append(item)
        elif not train and epoch:
            self.loss_dict['vepoch'].append(item)
        elif not train and not epoch:
            self.loss_dict['vbatch'].append(item)
        return item

# %% [markdown]
# ## TRAINING + VALIDATION

# %%
def one_epoch_train(model, train_loader, optimizer, loss_function, verbose=False):
    running_loss = 0
    last_loss = 0
    model.train()
    for i, data in enumerate(train_loader):
        inputs,labels, _ = data
        inputs,labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs) #, activation=F.relu)
        loss=loss_function(outputs, labels)
        loss.backward()
        optimizer.step() # fa update del parameter
        RMSEloss = np.sqrt(loss.item())
        running_loss += RMSEloss
        if verbose and i%10 ==0: print(f'Batch {i+1}/{len(train_loader)} - Loss: {RMSEloss:.3f}')

        model.log_the_loss(RMSEloss, epoch=False)
    epochmean_loss = running_loss / len(train_loader)
    #print(f'\nLoss: {epochmean_loss:.3f}')
    model.log_the_loss(epochmean_loss, epoch=True)
    last_loss = RMSEloss
    #print(f"Last loss: {last_loss:.3f}")
    return epochmean_loss



def one_epoch_eval(model, test_loader, loss_function, verbose=False):
    model.eval()
    running_validation_loss = 0.
   
    with torch.no_grad(): # deactivates gradient evaluation
        for i, vdata in enumerate(test_loader):
            inputs,labels, _ = vdata
            inputs,labels= inputs.to(device), labels.to(device)
            with torch.autocast('cuda'):
                outputs = model(inputs)#, activation=F.relu)
                loss = loss_function(outputs ,labels)
            RMSEloss = np.sqrt(loss.item())
            running_validation_loss +=RMSEloss
            model.log_the_loss(RMSEloss,epoch=False)
    mean_vloss=model.log_the_loss(running_validation_loss/len(test_loader),epoch=True)
    if verbose: print(f"Validation Loss: {mean_vloss:.3f}\n---")
    return mean_vloss

# %% [markdown]
# ## OPTUNA
# 

DS = GalaxyJungle('../data/training/training_solutions_rev1.csv', '../data/training/', transfs)
training, test = random_split(DS, [0.8, 0.2])

def objective(trial:optuna.Trial):
   
    ## Hyperspace
    num_conv_layers = 3
    #qui tuniamo il numero di filri, per layer più profondi ci vogliono più filtri (64-28 è consigliato per pattern astratti e combinazioni, mentre fino a 32 per dettagli locali) quindi proviamo (VGG usa fino a 512 per esempio).
    num_filters = [int(trial.suggest_int("num_filters_"+str(i), 6, 86 , step=8)) for i in range(num_conv_layers)]
    ## abbiamo numneurons1 e numn neurons2,se mettiamo un grid sampler o un random sampler con num_neurons e basta penso che lui provi diverse combinazioni
    num_neurons = trial.suggest_int("num_neurons", 30, 120, step=10) 
    ### abbiamo chiamato mode l'activation function nell'initialization dei pesi o la chiamiamo activation o FUNZIONEDIATTIVAZIONE così optuna poi iniializza in base a quello
    activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU"])
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW", 'RMSprop']) #AdamW è suggerito per CNN.
    learning_rate = trial.suggest_float("learning_rate", 5e-3, 5e-1, log=True) #log true cerca i valori in scala logaritmica
    momentum = trial.suggest_float("momentum", 0.4, 0.9, step=0.1) #per SGD
    # batch size da tunare?
    batch_size = 512
    epochs = 50
    loss_function = nn.MSELoss()
    
    ##### Training phase
    
    
    
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=8) 
    activation = getattr(nn, activation)
    #print(activation)
    model = GalaxyNet(num_conv_layers, num_filters, num_neurons, activation).to(device)
    if optimizer in ('SGD', 'RMSprop'): optimizer = getattr(optim, optimizer)(model.parameters(), lr=learning_rate, momentum = momentum)
    else: optimizer = getattr(optim, optimizer)(model.parameters(), lr=learning_rate)

    
    for epoch in range(0, epochs):
        #print(f'Training epoch {epoch}')
        one_epoch_train(model, train_loader, optimizer, loss_function, verbose=False)
        #print(f'Validation epoch {epoch}')
        epoch_last_val_loss = one_epoch_eval(model, test_loader, loss_function, verbose=False)
        trial.report(epoch_last_val_loss,epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    
    score = epoch_last_val_loss
    return score

# %%
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "3conv50epoch"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=50)

# %%



