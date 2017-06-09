import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from cnn_class import CNN
import torch.optim as optim
from utils import *

########################################################################
# 1. Defining hardware non-idealty

SI = 1 # reminder for sign bit
IL = 5  # IL = torch.IntTensor([4])   Not including the signed bit
FL = 14 # FL = torch.IntTensor([12])
nonideal_train = 1 #fixed point train
nonideal_inference = 1 #fixed point inference

en_ntv_inference = 1
en_ntv_train = 0
p_flip = 0.0001
flip_len = IL + FL + SI # flip all bits

save_model = 1
load_model = 0
en_plot = 0

EPOCH = 1

########################################################################
# 2. Preparing CIFAR10 data

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


########################################################################
# 3. defining non-idealty hooks

def fixed_point_hook(self, input, output):
    apply_format_inplace('FXP', output.data, IL, FL)

def fixed_point_back_hook(self, grad_in, grad_out):
    '''
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('grad_input tuple size: ', grad_in.__len__())
    #print('grad_output size:', grad_out[0].size())
    #print(type(grad_in))
    #print('grad_input size:', grad_in[0].size())
    #print(grad_in[0])

    if grad_in.__len__() > 1:
        if not (grad_in[0] is None):
            print('grad_input size:', grad_in[0].size())
        print('grad_input size:', grad_in[1].size())
        print('grad_input size:', grad_in[2].size())
    '''
    if grad_in.__len__() > 1:

            #grad_in[0] = Variable(None)
        apply_format_inplace('FXP', grad_in[1].data, IL, FL)  # weight
        apply_format_inplace('FXP', grad_in[2].data, IL, FL)  # bias
        if not (grad_in[0] is None):
            apply_format_inplace('FXP', grad_in[0].data, IL, FL)


        return grad_in


########################################################################
# 4. Create CNN instance and sgd optimizer
net = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)#, momentum=0.9)

juanji1 = list(net.conv1.parameters())
juanji2 = list(net.conv2.parameters())
linear1 = list(net.fc1.parameters())
linear2 = list(net.fc2.parameters())
linear3 = list(net.fc3.parameters())

# 4.1
if(nonideal_train):
    # add backward
    net.conv1.register_backward_hook(fixed_point_back_hook)
    net.conv2.register_backward_hook(fixed_point_back_hook)
    net.fc1.register_backward_hook(fixed_point_back_hook)
    net.fc2.register_backward_hook(fixed_point_back_hook)
    net.fc3.register_backward_hook(fixed_point_back_hook)
    net.activation.register_backward_hook(fixed_point_back_hook)
    # add forward
    net.conv1.register_forward_hook(fixed_point_hook)
    net.conv2.register_forward_hook(fixed_point_hook)
    net.fc1.register_forward_hook(fixed_point_hook)
    net.fc2.register_forward_hook(fixed_point_hook)
    net.fc3.register_forward_hook(fixed_point_hook)
    net.activation.register_forward_hook(fixed_point_hook)


# 4.2 weight init
# BLA BLA BLA

########################################################################
# 5. Train the network

for epoch in range(EPOCH):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if nonideal_train:
            apply_format_inplace('FXP', juanji1[0].data, IL, FL)  # weight regulation
            apply_format_inplace('FXP', juanji1[1].data, IL, FL)  # bias regulation
            apply_format_inplace('FXP', juanji2[0].data, IL, FL)  # weight regulation
            apply_format_inplace('FXP', juanji2[1].data, IL, FL)  # bias regulation
            apply_format_inplace('FXP', linear1[0].data, IL, FL)  # weight regulation
            apply_format_inplace('FXP', linear1[1].data, IL, FL)  # bias regulation
            apply_format_inplace('FXP', linear2[0].data, IL, FL)  # weight regulation
            apply_format_inplace('FXP', linear2[1].data, IL, FL)  # bias regulation
            apply_format_inplace('FXP', linear3[0].data, IL, FL)  # weight regulation
            apply_format_inplace('FXP', linear3[1].data, IL, FL)  # bias regulation





        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

if save_model:
    print('Saving model......')
    torch.save(net, 'cnn_cifar10_model.pkl')
    #>>>>>>>>Test neural network performance

if load_model:
    print('Loading model......')
    net = torch.load('cnn_cifar10_model.pkl')

########################################################################
# 6. Test the network
#
#
# 6.1 dump the handle from trained network layer
juanji1 = list(net.conv1.parameters())
juanji2 = list(net.conv2.parameters())
linear1 = list(net.fc1.parameters())
linear2 = list(net.fc2.parameters())
linear3 = list(net.fc3.parameters())


# 6.2
if nonideal_inference:
    apply_format_inplace('FXP',juanji1[0].data,IL,FL) # weight regulation
    apply_format_inplace('FXP', juanji1[1].data, IL, FL) # bias regulation
    apply_format_inplace('FXP',juanji2[0].data,IL,FL) # weight regulation
    apply_format_inplace('FXP', juanji2[1].data, IL, FL) # bias regulation
    apply_format_inplace('FXP',linear1[0].data,IL,FL)    # weight regulation
    apply_format_inplace('FXP', linear1[1].data, IL, FL) # bias regulation
    apply_format_inplace('FXP',linear2[0].data,IL,FL)    # weight regulation
    apply_format_inplace('FXP', linear2[1].data, IL, FL) # bias regulation
    apply_format_inplace('FXP',linear3[0].data,IL,FL)    # weight regulation
    apply_format_inplace('FXP', linear3[1].data, IL, FL) # bias regulation



if nonideal_inference and (not nonideal_train ):
    net.conv1.register_forward_hook(fixed_point_hook)
    net.conv2.register_forward_hook(fixed_point_hook)
    net.fc1.register_forward_hook(fixed_point_hook)
    net.fc2.register_forward_hook(fixed_point_hook)
    net.fc3.register_forward_hook(fixed_point_hook)
    net.activation.register_forward_hook(fixed_point_hook)


correct = 0
total = 0
for data in testloader:
    images, labels = data
    if nonideal_inference:
        apply_format_inplace('FXP', images, IL, FL)  # input regulation
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))