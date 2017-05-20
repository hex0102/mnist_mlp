import torch
from torch.autograd import Variable


class Formatting(torch.autograd.Function):

    def foward(self, x, format, IL,FL):
        x = apply_format(format,x,IL,FL);
        return x



    def backward(self, *grad_output):
        grad_input = grad_output.clone()
        return grad_input


def apply_format(format, X, IL, FL):
    if format == 'FXP':
        return fixed_point(X, IL,FL)
    elif format == "FLP":
        return X


def fixed_point(x, IL, FL):
    return x

def overflow(vector, IL, FL):
    pass