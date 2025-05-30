from tqdm import tqdm
import torch
import os
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import zookeeper as zk
import pickle


# Set if you want to use rgb/mappy
rgb = False
mappy = False
epoch_start = 1
epochs =1001


# Transformation to training images
transfs = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(324),
    transforms.Resize(128),
    transforms.RandomRotation(180)
    ])

# Transformation to validation images
test_transfs = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(324),
    transforms.Resize(128),
    ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('spada e s' + str(device))


class GalaxyJungle(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, mappy=False, is_rgb=False):
        self.rgb = is_rgb
        self.mappy = mappy
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        
    
    def __len__(self): return (self.img_labels).shape[0]

    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0])) + '.jpg'
        image = Image.open(img_path)
        if not self.rgb: image = image.convert('L')
        if self.transform: image = self.transform(image)
        
        label = self.img_labels.iloc[idx, 1:]        
        label = torch.tensor(label.values, dtype=torch.float32)
        if self.mappy: label = zk.mappy(label)

        gal_id = self.img_labels.iloc[idx, 0]

        return image, label, gal_id



# Tweak the class as you need, you should not change the mappy and rgb settings, would have probably been clearer if we created GalaxyNet subclasses...
# PC
class GalaxyNet(nn.Module):
    def __init__(self, activation, initialization=False, mappy=False, is_rgb=False):
        super().__init__()
        
        self.mappy = mappy
        rgb = 3 if is_rgb else 1
        input_size = 128
        num_labels = 37
        self.loss_dict = {'batch' : [], 'epoch' : [], 'vbatch' : [], 'vepoch' : []}
        self.activation = activation
        
        self.convs = nn.Sequential(
            nn.Conv2d(rgb, 64, 3, bias=False),
            self.activation(),
            nn.BatchNorm2d(64),
            
            nn.MaxPool2d(2)
            )

        for layer in self.convs:
            if layer.__class__.__name__ == 'Conv2d': input_size = zk.convool_size(input_size, 3, 1, 'same' if layer.padding == 'same' else 0)
            elif layer.__class__.__name__ == 'MaxPool2d': input_size = zk.convool_size(input_size, 2, 2)
        if input_size < 2: raise ValueError('You shrank too much dude.')
        print(f'Convs output size: {input_size}')

        input_linear = 64 * input_size * input_size
        
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
        
        for layer in self.convs: 
            if layer.__class__.__name__ == 'Conv2d': nn.init.kaiming_normal_(layer.weight, a=a, nonlinearity=nonlin)        

        for i in (1, -1): nn.init.constant_(self.fc[i].bias, 0)
 
        nn.init.kaiming_normal_(self.fc[1].weight, a=a, nonlinearity=nonlin)
        nn.init.xavier_uniform_(self.fc[-1].weight)      
        

    def log_the_loss(self, item,epoch=False):
        train = self.__getstate__()['training']
        if epoch and train:
            self.loss_dict['epoch'].append(item)
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
        optimizer.step()
        RMSEloss = np.sqrt(loss.item())
        running_loss += RMSEloss
        if verbose and i % 10 == 0: print(f'Batch {i + 1}/{len(train_loader)} - Loss: {RMSEloss:.3f}')
        model.log_the_loss(RMSEloss, epoch=False)
    epochmean_loss = running_loss / len(train_loader)
    print(f'\nLoss: {epochmean_loss}')
    model.log_the_loss(epochmean_loss, epoch=True)
    last_loss = RMSEloss
    print(f"Last loss: {last_loss}")
    return epochmean_loss


def one_epoch_eval(model, test_loader, loss_function):
    model.eval()
    running_validation_loss = 0.
    with torch.no_grad():
        for i, vdata in tqdm(enumerate(test_loader)):
            inputs,labels, _ = vdata
            inputs,labels= inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs ,labels)
            RMSEloss = np.sqrt(loss.item())
            running_validation_loss += RMSEloss
            model.log_the_loss(RMSEloss,epoch=False)
    mean_vloss=model.log_the_loss(running_validation_loss / len(test_loader),epoch=True)
    print(f"Validation Loss: {mean_vloss}\n---")
    return mean_vloss




training = GalaxyJungle('../data/training/training_solutions_rev1.csv', '../data/training/', transfs, mappy=mappy, is_rgb=rgb)
test = GalaxyJungle('../data/validation/validation_solutions_rev1.csv', '../data/validation/', test_transfs, mappy=mappy, is_rgb=rgb)

train_loader = DataLoader(training, batch_size=32, shuffle=True, num_workers=os.cpu_count())
test_loader = DataLoader(test, batch_size=32, shuffle=False, num_workers=os.cpu_count())    

model = GalaxyNet(nn.ReLU, initialization=False, mappy=mappy, is_rgb=rgb).to(device)
optimizer = optim.SGD(model.parameters(), lr=.002, momentum=.6)
loss_function = nn.MSELoss()

### to resume training uncomment next lines (and edit with last epoch num)
# loader = torch.load('model_optim_{}.pt', weights_only=True)
# model.load_state_dict(loader['model_state_dict'])
# optimizer.load_state_dict(loader['optimizer_state_dict'])

#with open('loss_{}.pickle', 'rb') as fin: losses = pickle.load(fin)
#for key in losses.keys():
#    for item in losses[key]: model.loss_dict[key].append(item)


for epoch in range(epoch_start, epochs):
    print(f'Training epoch {epoch}')
    one_epoch_train(model, train_loader, optimizer, loss_function)

    print(f'Validation epoch {epoch}')
    one_epoch_eval(model, test_loader, loss_function)
    
    if epoch % 10 == 0 or epoch == epochs - 1: 
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, f'model_optim_{epoch}.pt')

        with open(f'loss_{epoch}.pickle', 'wb') as fout: pickle.dump(model.loss_dict, fout, protocol=-1)
