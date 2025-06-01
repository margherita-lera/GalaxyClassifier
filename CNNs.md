# List of architectures

 Name | Layers | Best Loss | Trials | Epochs | Mappy | Authour 
 :---: | :---: | :---: | :---: | :---: | :---: | :---:
 JAGZoo | 5 convs + 1 fc | 0.09157407635789411 | 23 | 50 | N | Gio 
 NSC | 5 conv + 1 fc | 0.06371325288077401  | 40 | 50 | Y | Marghe
 PC | 1 conv + 1 fc | 0.12687206503096748 | 16 | 50 | N |Gio
 PADel | 7 conv + 1 fc |  0.08708066615417723 | 12 | 50 | N | Gio
 PADel_map | 7 conv + 1 fc |  0.06129642133313233 | 42 | 50 | Y | Gio
 Skynet | 5 conv + 2 fc |  | 30 | 50 | N | Gab
 3conv | 3 conv + 1 fc | 0.13609738523046297 | 67 | 50 | N | Gigi
 VGG | VGG | 0.08338175853681867 | 30 | 50 | N | Chad


# Inside the Architectures
(The layers show `ACTIVATION` instead of the activation function as it was an optunable parameter, though we could put the best activation).


## JAGZoo

 ```GalaxyNet(
  (convs): Sequential(
    (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (1): ACTIVATION
    (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (5): ACTIVATION
    (6): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (9): ACTIVATION
    (10): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (12): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (13): ACTIVATION
    (14): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (16): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (17): ACTIVATION
    (18): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (19): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=512, out_features=100, bias=True)
    (2): ACTIVATION
    (3): Linear(in_features=100, out_features=37, bias=True)
  )
)
```

I tried to go deeper in the convolutional layers. It seems that the net has learnt better than previous results with fewer layers.


## NSC

```GalaxyNet(
  (convs): Sequential(
    (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (9): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
    (11): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (12): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (13): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (14): ReLU()
    (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (16): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (17): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (18): ReLU()
    (19): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=512, out_features=100, bias=True)
    (2): ReLU()
    (3): Linear(in_features=100, out_features=37, bias=True)
  )
)
```
The net should be the same as JAGZoo with the batchnorm put before the activation layer. Also MAPPING included.


## PC

```
GalaxyNet(
  (convs): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (1): ACTIVATION
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=508032, out_features=100, bias=True)
    (2): ACTIVATION
    (3): Linear(in_features=100, out_features=37, bias=True)
  )
)
```

The almost SAP configuration. Results are not that good yet, which is reassuring.


## PADel

```
GalaxyNet(
  (convs): Sequential(
    (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=same, bias=False)
    (1): ReLU()
    (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=same, bias=False)
    (4): ReLU()
    (5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (7): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=same, bias=False)
    (8): ReLU()
    (9): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=same, bias=False)
    (11): ReLU()
    (12): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (14): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (15): ReLU()
    (16): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (17): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (18): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (19): ReLU()
    (20): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (21): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (22): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (23): ReLU()
    (24): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (25): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=1024, out_features=100, bias=True)
    (2): ReLU()
    (3): Linear(in_features=100, out_features=37, bias=True)
  )
)

```

I want to figure out if padding is as useless as they say.

## Skynet

```
Convs output size: 4
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
├─Sequential: 1-1                        --
|    └─Conv2d: 2-1                       2,496
|    └─ReLU: 2-2                         --
|    └─MaxPool2d: 2-3                    --
|    └─Conv2d: 2-4                       307,328
|    └─ReLU: 2-5                         --
|    └─MaxPool2d: 2-6                    --
|    └─Conv2d: 2-7                       409,728
|    └─ReLU: 2-8                         --
|    └─Conv2d: 2-9                       409,728
|    └─ReLU: 2-10                        --
|    └─Conv2d: 2-11                      819,456
|    └─ReLU: 2-12                        --
|    └─MaxPool2d: 2-13                   --
├─Sequential: 1-2                        --
|    └─Flatten: 2-14                     --
|    └─Dropout: 2-15                     --
|    └─Linear: 2-16                      8,390,656
|    └─ReLU: 2-17                        --
|    └─Linear: 2-18                      2,098,176
|    └─ReLU: 2-19                        --
|    └─Dropout: 2-20                     --
|    └─Linear: 2-21                      37,925
=================================================================
Total params: 12,475,493
Trainable params: 12,475,493
Non-trainable params: 0
=================================================================

```

## 3conv

``` GalaxyNet()
   for i in range(1,3):
            self.convs.append(nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=3, stride=2, bias=False))
            self.convs.append(self.activation())
            self.convs.append(nn.BatchNorm2d(num_filters[i]))
            self.convs.append(nn.MaxPool2d(kernel_size=2))
   self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.out_feature, num_neurons),
            self.activation(),
            nn.Linear(num_neurons, num_labels)
            )
   num_filters[0] : 6
   num_filters[1] : 38
   num_filters[2] : 30
   num_neurons : 80
   activation : "LeakyReLU"
   optimizer : SGD
   learning_rate : 0.12703853693077546
   momentum : 0.9
   batch_size : 512
   crop : 312
   resize : 256
```
   
   



