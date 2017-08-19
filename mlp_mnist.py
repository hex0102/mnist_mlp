import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as Data
from mlp_class import MLP
import matplotlib.pyplot as plt
from utils import *

import numpy as np


torch.manual_seed(4)


EPOCH = 0
BATCH_SIZE = 128
LR = 0.1
DOWNLOAD_MNIST = True
save_model = 0
load_model = 1
en_plot = 0

SI = 1 # reminder for sign bit
IL = 3  # IL = torch.IntTensor([4])   Not including the signed bit
FL = 3 # FL = torch.IntTensor([12])
nonideal_train = 0 #fixed point train
nonideal_inference = 1 #fixed point inference

#enabling ntv which increases bit flip probability
en_ntv_inference = 0
en_ntv_train = 0
p_flip = 0.001
flip_len = IL + FL - 1#IL + FL + SI # flip all bits IL + FL + SI

grads = [] #gradients are inside

#train_data = torchvision.datasets.MNIST(
#    root = './mnist/',
#    train=True,
#    transform=torchvision.transforms.ToTensor(),
#    download=DOWNLOAD_MNIST,
#)

train_data = torchvision.datasets.MNIST(
    root = './mnist/',
    train=True,
    transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    download=DOWNLOAD_MNIST,
)


test_data = torchvision.datasets.MNIST(root='./mnist/', train=False,transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1),volatile=True).type(torch.FloatTensor)
test_y = test_data.test_labels


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist/', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=False)



# defining hooks
def fixed_point_hook(self, input, output):
    '''
    if step % 50 == 0 and 'ReLU' == self.__class__.__name__:
        print('sum > 1 :', ((output.data)>1).sum())
        print('sum > 2 :', ((output.data) > 2).sum())
        print('sum > 3 :', ((output.data) > 3).sum())
        print('sum > 3.9 :', ((output.data) > 3.9).sum())
    '''

    # print('Inside ' + self.__class__.__name__ + ' forward')
    # print('Inside ' + self.__class__.__name__ + ' backward')
    apply_format_inplace('FXP', output.data, IL, FL)


def fixed_point_back_hook(self, grad_in, grad_out):

    #print('Inside ' + self.__class__.__name__ + ' backward')
    '''
    print('grad_output size:', grad_out[0].size())
    print('grad_input size:', grad_in[0].size())
    print('grad_input size:', grad_in[1].size())
    print('grad_input size:', grad_in[2].size())
    
    print('grad_output size:', grad_out[0].size())
    print(grad_out[0])
    print('grad_input size:', grad_in[0].size())
    print(grad_in[0])
    print('grad_input size:', grad_in[1].size())
    print('grad_input size:', grad_in[2].size())
    print('grad_input tuple size: ', grad_in.__len__())
    '''
    #grads[self.__class__.__name__] = grad_in[0].data;
    grads.append(grad_in[0].data);
    if grad_in.__len__() > 1:
        #print('grad_input size:', grad_in[1].size())
        apply_format_inplace('FXP', grad_in[0].data, IL, FL)
        apply_format_inplace('FXP', grad_in[1].data, IL, FL)#weight
        apply_format_inplace('FXP', grad_in[2].data, IL, FL)#bias

        #print(type(grad_in[0]))
        #print(type(grad_in[1]))
        #print(type(grad_in[2]))
        return grad_in



def mse_loss(x, y, size_average=True, per_element=True):
    #print(type(y))
    #print(y.data.size())
    #print(y.data.type())
    #print(y.data.size())
    y.data = y.data.unsqueeze(1)
    y_onehot = torch.FloatTensor(BATCH_S, 10)
    y_onehot.zero_()
    y_onehot.scatter_(1, y.data, 1)
    label = y_onehot
    label = Variable(label)
    loss_per_element = (x- label) ** 2
    if per_element:
        return loss_per_element.mean(1,keepdim=True)
    if size_average:
        return loss_per_element.mean()
    return loss_per_element.sum()



mlp=MLP()
optimizer = torch.optim.SGD(mlp.parameters(), lr=LR)#, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

#add backward hooker on each layer, to regulate the gradient to fixed point representation
if(nonideal_train):
    mlp.hidden1.register_backward_hook(fixed_point_back_hook)
    mlp.activation.register_backward_hook(fixed_point_back_hook)
    mlp.hidden2.register_backward_hook(fixed_point_back_hook)
    mlp.activation2.register_backward_hook(fixed_point_back_hook)
    mlp.out.register_backward_hook(fixed_point_back_hook)
    #add foward hooker on each layer, , to regulate the layer out to fixed point representation
    mlp.hidden1.register_forward_hook(fixed_point_hook)
    mlp.activation.register_forward_hook(fixed_point_hook)
    mlp.hidden2.register_forward_hook(fixed_point_hook)
    mlp.activation2.register_forward_hook(fixed_point_hook)
    mlp.out.register_forward_hook(fixed_point_hook)


#>>>>>>>>Dump neural network weight to list
linear = list(mlp.hidden1.parameters())
init.constant(linear[1].data,0)
linear2 = list(mlp.hidden2.parameters())
init.constant(linear2[1].data,0)
out = list(mlp.out.parameters())
init.constant(out[1].data,0)

init.normal(linear[0].data, mean=0, std=0.01)
init.normal(linear2[0].data, mean=0, std=0.01)
init.normal(out[0].data, mean=0, std=0.01)

train_loss_y = np.array([])
train_loss_x = np.array([])
i=0

for epoch in range(EPOCH):
    avg_loss = np.array([])
    for step, (x,y) in enumerate(train_loader):
        b_x = Variable(x,requires_grad = True)
        b_y = Variable(y)

        BATCH_S = b_x.size()[0]
        #print(BATCH_S)
        output = mlp(b_x)


        #loss = loss_func(output, b_y)
        loss = mse_loss(output,b_y)
        #print(loss.size())
        mlp.zero_grad()
        '''
        for i in range(BATCH_SIZE):
            output[i:i+1].backward(torch.ones(2,10),retain_variables=True)
            loss = loss_func(output,b_y)
            mlp.zero_grad()
        '''

        ind = torch.zeros(b_x.size()[0],1)
        ind[:,:] = 1

        optimizer.zero_grad()
        loss.backward(ind)
        optimizer.step()

        if nonideal_train:
            apply_format_inplace('FXP', linear[0].data, IL, FL)#weight
            apply_format_inplace('FXP', linear[1].data, IL, FL)#bias
            apply_format_inplace('FXP', linear2[0].data, IL, FL)#weight
            apply_format_inplace('FXP', linear2[1].data, IL, FL)#bias
            apply_format_inplace('FXP', out[0].data, IL, FL)#weight
            apply_format_inplace('FXP', out[1].data, IL, FL)#bias

        if en_ntv_train:
            apply_bitflip(linear[0].data, p_flip, IL, FL, flip_len)
            apply_bitflip(linear2[0].data, p_flip, IL, FL, flip_len)
            apply_bitflip(out[0].data, p_flip, IL, FL, flip_len)

        if step % 50 == 0:
            print('Epoch: ', epoch, '| train loss : %.8f' % loss.data[0][0])

        avg_loss = np.append(avg_loss, loss.data[0][0])


        '''
        if step % 120 == 0:
            print('printing!!!')
            train_loss_y = np.append(train_loss_y,loss.data[0])
            train_loss_x = np.append(train_loss_x,i)
            plt.plot(train_loss_x, train_loss_y, 'r-', lw=2)
            plt.pause(0.05)
            #plt.show(block=False)
            i = i + 1
        
        train_loss_x = train_loss_x.astype(int)
        '''

        '''
        if step % 50 == 0:
            test_output = mlp(test_x)
            pred_y = torch.max(test_output,1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y)/float(test_y.size(0))
            print('Epoch: ',epoch,'| train loss : %.4f' %loss.data[0], '| test accuracy: %.2f' % accuracy)
        '''

    #per epoch updating......
    if en_plot:
        train_loss_y = np.append(train_loss_y,np.mean(avg_loss))
        train_loss_x = np.append(train_loss_x, epoch)
        plt.semilogy(train_loss_x, train_loss_y, 'r-', lw=2)
        plt.xlabel('Epoch')
        plt.ylabel('Training error')
        plt.title('IL = 4, FL = 14')
        plt.grid(True)
        plt.pause(0.05)


plt.show()

if save_model:
    print('Saving model......')
    torch.save(mlp, 'mlp_backup2.pkl')
    #>>>>>>>>Test neural network performance

if load_model:
    print('Loading model......')
    mlp = torch.load('mlp_backup2.pkl')

#>>>>>>>>>>>>>>>>>>>>>>>>Inference
linear = list(mlp.hidden1.parameters())
linear2 = list(mlp.hidden2.parameters())
out = list(mlp.out.parameters())


#>>>>>>>>Regulate trained neural network weight into fixed point counterpart, then do inference
if nonideal_inference:
    apply_format_inplace('FXP',linear[0].data,IL,FL) # weight regulation
    apply_format_inplace('FXP', linear[1].data, IL, FL) # bias regulation
    apply_format_inplace('FXP',linear2[0].data,IL,FL) # weight regulation
    apply_format_inplace('FXP', linear2[1].data, IL, FL) # bias regulation
    apply_format_inplace('FXP',out[0].data,IL,FL)    # weight regulation
    apply_format_inplace('FXP', out[1].data, IL, FL) # bias regulation
    apply_format_inplace('FXP',test_x.data,IL,FL)    # input regulation
    #apply_bitflip(linear[0].data,0.5,3,10,13)



if en_ntv_inference:
    print("injecting bit flip fault model...")
    apply_bitflip(linear[0].data,p_flip,IL,FL,flip_len)
    apply_bitflip(linear2[0].data, p_flip, IL, FL, flip_len)
    apply_bitflip(out[0].data, p_flip, IL, FL, flip_len)
    #apply_bitflip(x,p,IL,FL,flip_length):


#print(type(test_x)) #<class 'torch.autograd.variable.Variable'>
#print(type(test_x.data)) #<class 'torch.FloatTensor'>


#>>>>>>>>add forward hooker on each layer if non-ideal inference
if nonideal_inference and (not nonideal_train ):
    #print('applied!!!')
    mlp.hidden1.register_forward_hook(fixed_point_hook)
    mlp.activation.register_forward_hook(fixed_point_hook)
    mlp.hidden2.register_forward_hook(fixed_point_hook)
    mlp.activation2.register_forward_hook(fixed_point_hook)
    mlp.out.register_forward_hook(fixed_point_hook)

#>>>>>>>> Testing!!!
#test_output = mlp(test_x)
#pred_y = torch.max(test_output, 1)[1].data.squeeze()
#accuracy = sum(pred_y == test_y)/float(test_y.size(0))
#print(accuracy, 'prediction accuracy!')


test_loss = 0
correct = 0
for data, target in test_loader:

    #data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    output = mlp(data)
    #test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

#test_loss /= len(test_loader.dataset)
print('\nTest set: Accuracy: (%.4f)\n', 100. * correct / len(test_loader.dataset))

print('End of testing!')
