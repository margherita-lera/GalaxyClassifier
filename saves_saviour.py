# This script is meant to be run in a directory containing all artifacts file with bad saving from torch.save
# Pay particular attention to the torch.save f parameter which is customised to add an extension indicating whether the object to save are state_dict or simple loss dict
# Not cool for marghe runs, it should be something like file[:-3] + EXTENSION This should remove the .pt part of the file name and add the new one.

import torch
import os

with os.scandir() as dir:
    for file in dir:
        loader = torch.load(file)
        torch.save({'model_state_dict': loader['model_state_dict'], 
                    'optim_state_dict' : loader['optim_state_dict']}, file + '.pt')
        torch.save(loader['losses'], file + '.pyd')