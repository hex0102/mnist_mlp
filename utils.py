import torch
import torchvision


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


def overflow(vector, IL, FL):
    pass