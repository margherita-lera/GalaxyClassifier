import pandas as pd
import numpy as np
import torch


def convool_size(input_size, kernel_size, stride, padding=0, dilation=1):
    '''
    Returns Conv2d and Max_pool output size for a square input tensor.
    '''
    if padding == 'same': output_size = input_size
    else: output_size = ((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    return output_size