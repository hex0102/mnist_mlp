import torch
import numpy as np


def restore_net(modelpath):
    # restore entire net1 to net2
    net = torch.load(modelpath)
    return net


def apply_format_inplace(format, X, IL, FL):
    if format == 'FXP':
        return fixed_point_inplace(X, IL, FL)
    elif format == "FLP":
        return X


def fixed_point_inplace(x, IL, FL):
    power = 2. ** FL
    MAX = (2.**(IL+FL))-1
    x.mul_(power)
    x.round_()
    x.clamp_(min=-MAX, max=MAX)
    x.div_(power)
    #x = torch.clamp(x,min=-MAX, max=MAX)
    #x = torch.div(x,power)
    return x


def apply_bitflip(x,p,IL,FL,flip_length):  #IL here means integer part not including(sign bit)
    temp = x.numpy()
    temp = temp*2. ** FL
    temp = temp.astype(int)
    changes = np.zeros([temp.shape[0],temp.shape[1]])
    flip_length = flip_length - 1 #  [1 4 14]  flip_length
    if flip_length != (IL+FL):
        for i in range(flip_length):
            value_mask=np.ones([temp.shape[0],temp.shape[1]]) * (2**(i-FL))
            zero_mask = np.bitwise_and(temp, 2**i)
            zero_mask[zero_mask>0] = -1
            zero_mask[zero_mask==0] = 1
            flip_mask = np.random.binomial(1, p, [temp.shape[0],temp.shape[1]])
            mask = flip_mask*zero_mask*value_mask
            changes = changes + mask
    else: #including the sign bit
        for i in range(flip_length-1):
            value_mask=np.ones([temp.shape[0],temp.shape[1]]) * (2**(i-FL))
            zero_mask = np.bitwise_and(temp, 2**i)  #this funcs considers signed representation, so no worry
            zero_mask[zero_mask > 0] = -1  # 1 -> 0  reduces the value
            zero_mask[zero_mask == 0] = 1  # 0 -> 1  increases the value
            flip_mask = np.random.binomial(1, p, [temp.shape[0],temp.shape[1]])
            mask = flip_mask*zero_mask*value_mask
            changes = changes + mask
        #for the sign bit | reducing and increasing is much different
        value_mask = np.ones([temp.shape[0], temp.shape[1]]) * (2 ** IL)
        zero_mask = np.bitwise_and(temp, 2 **flip_length)
        zero_mask[zero_mask > 0] = 1
        zero_mask[zero_mask == 0] = -1
        flip_mask = np.random.binomial(1, p, [temp.shape[0], temp.shape[1]])
        mask = flip_mask * zero_mask * value_mask
        changes = changes + mask
    changes = torch.from_numpy(changes)
    changes = changes.type(torch.FloatTensor)
    x.add_(changes)
    # considering the negative x
    print('worked!')



def overflow(vector, IL, FL):
    pass


