# List of architectures

 Name | Layers | Best Loss | Trials | Epochs | Authour 
 :---: | :---: | :---: | :---: | :---: | :---: 
 JAGZoo | 5 convs + 1 fc | 0.09157407635789411 | 23 | 50 | Gio 
 NSC | 5 conv + 1 fc | 0.09639266573163484  | 2 | 50 | Marghe
 PC | 1 conv + 1 fc | 0.12687206503096748 | 16 | 50 | Gio
 



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

```
NOT FOUND
```

(Not the author) The net should be the same as JAGZoo with the batchnorm put before the activation layer. More trials are required.


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
