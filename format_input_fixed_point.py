import torch
from torch.autograd import Variable


'''
class Formatting(torch.autograd.Function):

    def forward(self, x):
        format='FXP'
        IL=4
        FL=12
        x = apply_format(format,x,IL,FL)
        return x



    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


def apply_format(format, X, IL, FL):
    if format == 'FXP':
        return fixed_point(X, IL,FL)
    elif format == "FLP":
        return X


def fixed_point(x, IL, FL):
    power = 2. ** FL
    MAX = (2.**(IL+FL))-1
    x.mul_(power)
    x=x.round()
    x=x.clamp(min=-MAX, max=MAX)
    x.div_(power)
    #x = torch.clamp(x,min=-MAX, max=MAX)
    #x = torch.div(x,power)
    return x
'''

