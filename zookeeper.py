import pandas as pd
import torch
import numpy as np



def convool_size(input_dimension, kernel_size, stride, padding=0, dilation=1):
    '''
    Returns Conv2d and Max_pool output size for a square input tensor.
    '''
    return ((input_dimension + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1




def mappy2D(tensor,datafr=False):

    '''
    This function takes as input a 2D tensor with the columns as structured in the .csv label file.
    It returns a new 2D tensor with as many rows as in the input,
    and a new set of labels that refer to the independent probabilities of 17 different classes.
    To see the result and the classes' name in a Pandas Dataframe set the parameter 'datafr' to True. 
    '''
    device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device('cpu')
    #tensor=torch.tensor(tensor) if not isinstance(tensor, torch.Tensor) else tensor
    class1_3 = (tensor[:,0:3])#.unsqueeze(0)
    class3_2 = (tensor[:,5:7])#.unsqueeze(0)
    class4_2 = (tensor[:,7:9])#.unsqueeze(0))
    class5_4 = (tensor[:,9:13])#.unsqueeze(0))
    class7_3 = (tensor[:,15:18])#.unsqueeze(0))
    class9_3 = (tensor[:,25:28])#.unsqueeze(0))

    #normalizing class 4 and 5
    class4_2 = torch.where(class4_2.sum(dim=1, keepdim=True) != 0, 
                           class4_2 / class4_2.sum(dim=1, keepdim=True), 
                           class4_2)
    class5_4 = torch.where(class5_4.sum(dim=1, keepdim=True) != 0, 
                           class5_4 / class5_4.sum(dim=1, keepdim=True), 
                           class5_4)

    E0=class7_3[:,0]
    E3=class7_3[:,1]
    E6=class7_3[:,2]

    S0a_eon=class9_3[:,0]
    SB0a_eon=class9_3[:,1]
    Scd_eon=class9_3[:,2]

    SoB=class4_2[:,1]*class3_2[:,0]
    SoA=class4_2[:,1]*class3_2[:,1]

    SAa=class4_2[:,0]*class3_2[:,1]*class5_4[:,3]
    SAb=class4_2[:,0]*class3_2[:,1]*class5_4[:,2]
    SAc=class4_2[:,0]*class3_2[:,1]*class5_4[:,1]
    SAd=class4_2[:,0]*class3_2[:,1]*class5_4[:,0]

    SBa=class4_2[:,0]*class3_2[:,0]*class5_4[:,3]
    SBb=class4_2[:,0]*class3_2[:,0]*class5_4[:,2]
    SBc=class4_2[:,0]*class3_2[:,0]*class5_4[:,1]
    SBd=class4_2[:,0]*class3_2[:,0]*class5_4[:,0]

    A=class1_3[:,2]
    if datafr==True:x= pd.DataFrame({'E0':E0,
                     'E3': E3,
                     'E6': E6,
                     'S0a_eon': S0a_eon,
                     'SB0a_eon':SB0a_eon,
                     'Scd_eon': Scd_eon,
                     'SoB': SoB,
                     'SoA': SoA,
                     'SAa': SAa,
                     'SAb': SAb,
                     'SAc': SAc,
                     'SAd': SAd,
                     'SBa': SBa,
                     'SBb': SBb,
                     'SBc': SBc,
                     'SBd': SBd,
                     'A': A})
    else: x=torch.stack([E0,
                     E3,
                     E6,
                     S0a_eon,
                     SB0a_eon,
                     Scd_eon,
                     SoB,
                     SoA,
                     SAa,
                     SAb,
                     SAc,
                     SAd,
                     SBa,
                     SBb,
                     SBc,
                     SBd,
                     A],dim=1).to(device)
    return (x)
    
def mappy(tensor,datafr=False):

    '''
    This function takes as input a 1D tensor with the columns as structured in the .csv label file.
    It returns a new 1D tensor with a new set of labels that refer to the independent probabilities of 17 different classes.
    To see the result and the classes' name in a Pandas Dataframe set the parameter 'datafr' to True. 

    NB:if you do a Python list of tensors (like E0, SoA, etc.), when you return it to GalaxyNet.forward, 
    it doesn’t match the expected tensor behavior, and crucially:
    it does not inherit the device (CPU/GPU) of the model's forward pass. So you would get compatibility 
    errors if you try to use it in a model that expects a tensor.
    '''

    device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device('cpu')
    class1_3 = (tensor[0:3])
    class3_2 = (tensor[5:7])
    class4_2 = (tensor[7:9])
    class5_4 = (tensor[9:13])
    class7_3 = (tensor[15:18])
    class9_3 = (tensor[25:28])
    
    #normalizing class 4 and 5
    class4_2 = class4_2/class4_2.sum() if class4_2.sum() != 0 else class4_2
    class5_4 = class5_4/class5_4.sum() if class5_4.sum() != 0 else class5_4

    E0=class7_3[0]
    E3=class7_3[1]
    E6=class7_3[2]

    S0a_eon=class9_3[0]
    SB0a_eon=class9_3[1]
    Scd_eon=class9_3[2]

    SoB=class4_2[1]*class3_2[0]
    SoA=class4_2[1]*class3_2[1]

    SAa=class4_2[0]*class3_2[1]*class5_4[3]
    SAb=class4_2[0]*class3_2[1]*class5_4[2]
    SAc=class4_2[0]*class3_2[1]*class5_4[1]
    SAd=class4_2[0]*class3_2[1]*class5_4[0]
    
    SBa=class4_2[0]*class3_2[0]*class5_4[3]
    SBb=class4_2[0]*class3_2[0]*class5_4[2]
    SBc=class4_2[0]*class3_2[0]*class5_4[1]
    SBd=class4_2[0]*class3_2[0]*class5_4[0]

    A=class1_3[2]

    values = torch.stack([
        E0, E3, E6,
        S0a_eon, SB0a_eon, Scd_eon,
        SoB, SoA,
        SAa, SAb, SAc, SAd,
        SBa, SBb, SBc, SBd,
        A
    ]).to(device).detach()

    if datafr:
        return pd.DataFrame([dict(zip([
            'E0', 'E3', 'E6',
            'S0a_eon', 'SB0a_eon', 'Scd_eon',
            'SoB', 'SoA',
            'SAa', 'SAb', 'SAc', 'SAd',
            'SBa', 'SBb', 'SBc', 'SBd',
            'A'
        ], values.cpu().numpy()))])
    
    return values



def mappy_df(tensor):

    '''
    This function takes a 2d Dataframe and does all the same again!
    '''
    
    class1_3 = np.array(tensor.iloc[:,1:4])
    class3_2 = np.array(tensor.iloc[:,6:8])
    class4_2 = np.array(tensor.iloc[:,8:10])
    class5_4 = np.array(tensor.iloc[:,10:14])
    class7_3 = np.array(tensor.iloc[:,16:19])
    class9_3 = np.array(tensor.iloc[:,26:29])
    
    #normalizing class 4 and 5
    class4_2 = np.where(class4_2.sum(axis=1, keepdims=True) != 0, 
                           class4_2 / class4_2.sum(axis=1, keepdims=True), 
                           class4_2)
    class5_4 = np.where(class5_4.sum(axis=1, keepdims=True) != 0, 
                           class5_4 / class5_4.sum(axis=1, keepdims=True), 
                           class5_4)
    E0=class7_3[:,0]
    E3=class7_3[:,1]
    E6=class7_3[:,2]

    S0a_eon=class9_3[:,0]
    SB0a_eon=class9_3[:,1]
    Scd_eon=class9_3[:,2]

    SoB=class4_2[:,1]*class3_2[:,0]
    SoA=class4_2[:,1]*class3_2[:,1]

    SAa=class4_2[:,0]*class3_2[:,1]*class5_4[:,3]
    SAb=class4_2[:,0]*class3_2[:,1]*class5_4[:,2]
    SAc=class4_2[:,0]*class3_2[:,1]*class5_4[:,1]
    SAd=class4_2[:,0]*class3_2[:,1]*class5_4[:,0]

    SBa=class4_2[:,0]*class3_2[:,0]*class5_4[:,3]
    SBb=class4_2[:,0]*class3_2[:,0]*class5_4[:,2]
    SBc=class4_2[:,0]*class3_2[:,0]*class5_4[:,1]
    SBd=class4_2[:,0]*class3_2[:,0]*class5_4[:,0]

    A=class1_3[:,2]

    x= pd.DataFrame({'GalaxyID':tensor['GalaxyID'],
                     'E0':E0,
                     'E3': E3,
                     'E6': E6,
                     'S0a_eon': S0a_eon,
                     'SB0a_eon':SB0a_eon,
                     'Scd_eon': Scd_eon,
                     'SoB': SoB,
                     'SoA': SoA,
                     'SAa': SAa,
                     'SAb': SAb,
                     'SAc': SAc,
                     'SAd': SAd,
                     'SBa': SBa,
                     'SBb': SBb,
                     'SBc': SBc,
                     'SBd': SBd,
                     'A': A})
    
    return x