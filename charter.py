import optuna
from tqdm import tqdm
import logging
import sys
import torch
import os
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import random_split, Dataset, DataLoader
from PIL import Image
import zookeeper as zk  # convool_size & mappy

rgb = False
mappy = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class GalaxyJungle(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, mappy=False, is_rgb=False):
        self.rgb = is_rgb
        self.mappy = mappy
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return (self.img_labels).shape[0]

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0])) + '.jpg'
        #retrieves the image
        image = Image.open(img_path)
        if not self.rgb: image = image.convert('L')
        if self.transform:
            image = self.transform(image)#.type(torch.float16)
        #retrieves corresponding label
        label = self.img_labels.iloc[idx, 1:]        
        label = torch.tensor(label.values, dtype=torch.float32)
        if self.mappy: label = zk.mappy(label)

        return image, label, self.img_labels.iloc[idx, 0]
    

transfs = transforms.Compose([
    transforms.ToTensor(), # Riscala le immagini tra 0 e 1
    transforms.CenterCrop(324),
    transforms.Resize(128),
    transforms.RandomRotation(180)
    ])



class GalaxyNet(nn.Module):
    def __init__(self, activation, initialization=False, mappy=False, is_rgb=False):
        super().__init__()
        
        self.mappy = mappy
        rgb = 3 if is_rgb else 1
        input_size = 128
        num_labels = 37
        
        self.loss_dict = {'batch' : [], 'epoch' : [], 'vbatch' : [], 'vepoch' : []}
        self.activation = activation

        
        ## convolutional layers
        self.convs = nn.Sequential(
            nn.Conv2d(rgb, 16, 3, padding='same', bias=False),
            self.activation(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, 3, padding='same', bias=False),
            self.activation(),
            nn.BatchNorm2d(16),
            
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding='same', bias=False),
            self.activation(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, 3, padding='same', bias=False),
            self.activation(),
            nn.BatchNorm2d(32),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, bias=False),
            self.activation(),
            nn.BatchNorm2d(64),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, bias=False),
            self.activation(),
            nn.BatchNorm2d(128),

            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, bias=False),
            self.activation(),
            nn.BatchNorm2d(256),

            nn.MaxPool2d(2)
            )

        for layer in self.convs:
            if layer.__class__.__name__ == 'Conv2d': input_size = zk.convool_size(input_size, 3, 1, 'same' if layer.padding == 'same' else 0)
            elif layer.__class__.__name__ == 'MaxPool2d': input_size = zk.convool_size(input_size, 2, 2)

        if input_size < 2: raise ValueError('You shrank too much dude.')
        print(f'Convs output size: {input_size}')

        input_linear = 256 * input_size * input_size
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_linear, 100),
            self.activation(),
            nn.Linear(100, num_labels)
            )

        if initialization: self.init_weights()



    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x)
        if self.mappy: x = zk.mappy2D(x)
        return x



    def init_weights(self):
        if self.activation == nn.ReLU:
            nonlin = 'relu'
            a = 0
        elif self.activation == nn.LeakyReLU:
            nonlin = 'leaky_relu'
            a = .01
        
        # Init convolutional parameters
        for layer in self.convs: 
            if layer.__class__.__name__ == 'Conv2d': nn.init.kaiming_normal_(layer.weight, a=a, nonlinearity=nonlin)
        

        # Init linear parameters
        for i in (1, -1): nn.init.constant_(self.fc[i].bias, 0)
        nn.init.kaiming_normal_(self.fc[1].weight, a=a, nonlinearity=nonlin)
        nn.init.xavier_uniform_(self.fc[-1].weight)      
        


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


def one_epoch_train(model, train_loader, optimizer, loss_function, verbose=False):
    running_loss = 0
    model.train()
    for i, data in tqdm(enumerate(train_loader)):
        inputs,labels, _ = data
        inputs,labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss=loss_function(outputs, labels)
        loss.backward()
        optimizer.step() # fa update del parameter
        RMSEloss = np.sqrt(loss.item())
        running_loss += RMSEloss
        if verbose and i%10 ==0: print(f'Batch {i+1}/{len(train_loader)} - Loss: {RMSEloss:.3f}')

        model.log_the_loss(RMSEloss, epoch=False)
    epochmean_loss = running_loss / len(train_loader)
    print(f'\nLoss: {epochmean_loss}')
    model.log_the_loss(epochmean_loss, epoch=True)
    last_loss = RMSEloss
    print(f"Last loss: {last_loss}")
    return epochmean_loss


def one_epoch_eval(model, test_loader, loss_function, verbose=False):
    model.eval()
    running_validation_loss = 0.
   
    with torch.no_grad(): # deactivates gradient evaluation
        for i, vdata in tqdm(enumerate(test_loader)):
            inputs,labels, _ = vdata
            inputs,labels= inputs.to(device), labels.to(device)
            outputs = model(inputs)#, activation=F.relu)
            loss = loss_function(outputs ,labels)
            RMSEloss = np.sqrt(loss.item())
            running_validation_loss +=RMSEloss
            model.log_the_loss(RMSEloss,epoch=False)
    mean_vloss=model.log_the_loss(running_validation_loss / len(test_loader),epoch=True)
    print(f"Validation Loss: {mean_vloss}\n---")
    return mean_vloss



DS = GalaxyJungle('../data/training/training_solutions_rev1.csv', '../data/training/', transfs, mappy=mappy, is_rgb=rgb)
training, test = random_split(DS, [.8, .2])

artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path='./artifacts')

def objective(trial:optuna.Trial):
    epochs = 50
    loss_function = nn.MSELoss()
    train_loader = DataLoader(training, batch_size=32, shuffle=True, num_workers=os.cpu_count())
    test_loader = DataLoader(test, batch_size=32, shuffle=False, num_workers=os.cpu_count())    
    
    # Trial choices
    activation = trial.suggest_categorical("activation", ['ReLU', 'LeakyReLU'])
    optimizer = trial.suggest_categorical("optimizer", ['Adam', 'SGD', 'AdamW', 'RMSprop', 'Adagrad', 'NAdam']) #AdamW è suggerito per CNN.
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True) #log true cerca i valori in scala logaritmica
    initialization = trial.suggest_categorical('init weight', [True, False])
    
    # Training phase
    activation = getattr(nn, activation)
    model = GalaxyNet(activation, initialization, mappy=mappy, is_rgb=rgb).to(device)
    if optimizer in ('SGD', "RMSprop"):
        momentum = trial.suggest_float("momentum", 0.3, 0.9, step=0.1)
        optimizer = getattr(optim, optimizer)(model.parameters(), lr=learning_rate, momentum = momentum)
    else: optimizer = getattr(optim, optimizer)(model.parameters(), lr=learning_rate)
    
    
    for epoch in range(epochs):
        print(f'Training epoch {epoch}')
        one_epoch_train(model, train_loader, optimizer, loss_function, verbose=False)

        print(f'Validation epoch {epoch}')
        epoch_last_val_loss = one_epoch_eval(model, test_loader, loss_function, verbose=False)
        trial.report(epoch_last_val_loss, epoch)


        if trial.should_prune(): raise optuna.TrialPruned()

    torch.save({
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'losses': model.loss_dict
    }, 'model.pt')

    art_id = optuna.artifacts.upload_artifact(artifact_store=artifact_store, file_path='model.pt', study_or_trial=trial.study)
    trial.set_user_attr('artifact_id', art_id)

    
    return epoch_last_val_loss


optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "PADel_map"
storage_name = f"sqlite:///CanOTuna.db"
study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=1)