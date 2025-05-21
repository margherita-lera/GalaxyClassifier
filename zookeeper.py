import pandas as pd
import torch



def convool_size(input_size, kernel_size, stride, padding=0, dilation=1):
    '''
    Returns Conv2d and Max_pool output size for a square input tensor.
    '''
    if str(padding) == 'same': output_size = input_size
    else: output_size = ((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    return output_size




def mappy(tensor,datafr=False):

    '''
    This function takes as input a 2D tensor with the columns as structured in the .csv label file.
    It returns a new 2D tensor with as many rows as in the input,
    and a new set of labels that refer to the independent probabilities of 17 different classes.
    To see the result and the classes' name in a Pandas Dataframe set the parameter 'datafr' to True. 
    '''
    class1_3 = (tensor[:,0:3])
    class3_2 = (tensor[:,5:7])
    class4_2 = (tensor[:,7:9])
    class5_4 = (tensor[:,9:13])
    class7_3 = (tensor[:,15:18])
    class9_3 = (tensor[:,25:28])

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
                     A],dim=1)
    return (x)
    
