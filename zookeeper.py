def convool_size(input_dimension, padding, dilation, kernel_size, stride):
    '''
    Returns Conv2d and Max_pool output size for a square input tensor.
    '''
    return ((input_dimension + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1

