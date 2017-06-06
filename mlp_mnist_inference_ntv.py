import torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as Data
from mlp_class import MLP
import matplotlib.pyplot as plt
from utils import *

load_model = 1
BATCH_SIZE = 100
LR = 0.1
DOWNLOAD_MNIST = True
epoch_test = 10


SI = 1 # reminder for sign bit
IL = 5  # IL = torch.IntTensor([4])   Not including the signed bit
FL = 14 # FL = torch.IntTensor([12])

train_data = torchvision.datasets.MNIST(
    root = './mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]




def fixed_point_hook(self, input, output):
    apply_format_inplace('FXP', output.data, IL, FL)


def fixed_point_back_hook(self, grad_in, grad_out):
    if grad_in.__len__() > 1:
        apply_format_inplace('FXP', grad_in[0].data, IL, FL)
        apply_format_inplace('FXP', grad_in[1].data, IL, FL)#weight
        apply_format_inplace('FXP', grad_in[2].data, IL, FL)#bias
        return grad_in


flip_range = [20,18,16,12,8]
p_flip_range = [0.001,0.00001,0.0000001]



for m in range(len(flip_range)):
    for n in range(len(p_flip_range)):
        flip_len = flip_range[m]
        p_flip = p_flip_range[n]
        temp = 0
        for i in range(epoch_test):
            if load_model:
                print('Loading model......')
                mlp = torch.load('mlp_mnist_10epoch.pkl')

            linear = list(mlp.hidden1.parameters())
            linear2 = list(mlp.hidden2.parameters())
            out = list(mlp.out.parameters())

            apply_bitflip(linear[0].data, p_flip, IL, FL, flip_len)
            apply_bitflip(linear2[0].data, p_flip, IL, FL, flip_len)
            apply_bitflip(out[0].data, p_flip, IL, FL, flip_len)

            test_output = mlp(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y)/float(test_y.size(0))
            temp = temp+accuracy
            print(accuracy, 'prediction accuracy!')
            print('End of testing!')

        temp = temp/epoch_test