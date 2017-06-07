#this is used for ntv induced bit flip  in non-ideal training

import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as Data
from mlp_class import MLP
import matplotlib.pyplot as plt
from utils import *

import numpy as np

#torch.manual_seed(2)



EPOCH = 12
BATCH_SIZE = 100
LR = 0.1
DOWNLOAD_MNIST = True
save_model = 0
load_model = 0
en_plot = 0

SI = 1  # reminder for sign bit
IL = 5  # IL = torch.IntTensor([4])   Not including the signed bit
FL = 14  # FL = torch.IntTensor([12])
nonideal_train = 1  # fixed point train
nonideal_inference = 1  # fixed point inference

# enabling ntv which increases bit flip probability
en_ntv_inference = 1
en_ntv_train = 0
p_flip = 0.0001
flip_len = IL + FL + SI  # flip all bits

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000] / 255.
test_y = test_data.test_labels[:2000]


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
    '''
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('grad_output size:', grad_out[0].size())
    print(grad_out[0])
    print('grad_input size:', grad_in[0].size())
    print(grad_in[0])
    print('grad_input size:', grad_in[1].size())
    print('grad_input size:', grad_in[2].size())
    print('grad_input tuple size: ', grad_in.__len__())
    '''
    if grad_in.__len__() > 1:
        # print('grad_input size:', grad_in[1].size())
        apply_format_inplace('FXP', grad_in[0].data, IL, FL)
        apply_format_inplace('FXP', grad_in[1].data, IL, FL)  # weight
        apply_format_inplace('FXP', grad_in[2].data, IL, FL)  # bias
        # print(type(grad_in[0]))
        # print(type(grad_in[1]))
        # print(type(grad_in[2]))
        return grad_in





has_mat = 0
flip_range = [20,18,16,12,8]
p_flip_range = [0.001,0.0001,0.00001,0.000001,0.0000001]

#flip_range = [20,18]
#p_flip_range = [0.001,0.0001]
mc_epoch = 5

loss_mat = np.zeros([len(flip_range), len(p_flip_range), mc_epoch, EPOCH],dtype=float)



for m in range(len(flip_range)):
    for n in range(len(p_flip_range)):
        for cc in range(mc_epoch):
            mlp = MLP()
            optimizer = torch.optim.SGD(mlp.parameters(), lr=LR)  # , momentum=0.9)
            loss_func = nn.CrossEntropyLoss()

            # add backward hooker on each layer, to regulate the gradient to fixed point representation
            if (nonideal_train):
                mlp.hidden1.register_backward_hook(fixed_point_back_hook)
                mlp.activation.register_backward_hook(fixed_point_back_hook)
                mlp.hidden2.register_backward_hook(fixed_point_back_hook)
                mlp.activation2.register_backward_hook(fixed_point_back_hook)
                mlp.out.register_backward_hook(fixed_point_back_hook)
                # add foward hooker on each layer, , to regulate the layer out to fixed point representation
                mlp.hidden1.register_forward_hook(fixed_point_hook)
                mlp.activation.register_forward_hook(fixed_point_hook)
                mlp.hidden2.register_forward_hook(fixed_point_hook)
                mlp.activation2.register_forward_hook(fixed_point_hook)
                mlp.out.register_forward_hook(fixed_point_hook)

            # >>>>>>>>Dump neural network weight to list
            linear = list(mlp.hidden1.parameters())
            init.constant(linear[1].data, 0)
            linear2 = list(mlp.hidden2.parameters())
            init.constant(linear2[1].data, 0)
            out = list(mlp.out.parameters())
            init.constant(out[1].data, 0)

            init.normal(linear[0].data, mean=0, std=0.01)
            init.normal(linear2[0].data, mean=0, std=0.01)
            init.normal(out[0].data, mean=0, std=0.01)

            train_loss_y = np.array([])
            train_loss_x = np.array([])

            flip_len = flip_range[m]
            p_flip = p_flip_range[n]

            for epoch in range(EPOCH):
                avg_loss = np.array([])
                for step, (x, y) in enumerate(train_loader):
                    b_x = Variable(x, requires_grad=True)
                    b_y = Variable(y)

                    output = mlp(b_x)
                    loss = loss_func(output, b_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if nonideal_train:
                        apply_format_inplace('FXP', linear[0].data, IL, FL)  # weight
                        apply_format_inplace('FXP', linear[1].data, IL, FL)  # bias
                        apply_format_inplace('FXP', linear2[0].data, IL, FL)  # weight
                        apply_format_inplace('FXP', linear2[1].data, IL, FL)  # bias
                        apply_format_inplace('FXP', out[0].data, IL, FL)  # weight
                        apply_format_inplace('FXP', out[1].data, IL, FL)  # bias


                    if en_ntv_train:
                        apply_bitflip(linear[0].data, p_flip, IL, FL, flip_len)
                        apply_bitflip(linear2[0].data, p_flip, IL, FL, flip_len)
                        apply_bitflip(out[0].data, p_flip, IL, FL, flip_len)

                    if step % 50 == 0:
                        print('Epoch: ', epoch, '| train loss : %.4f' % loss.data[0])


                    avg_loss = np.append(avg_loss, loss.data[0])

                loss_mat[m, n, cc, epoch] = np.mean(avg_loss)
                # per epoch updating......
                if en_plot:
                    train_loss_y = np.append(train_loss_y, np.mean(avg_loss))
                    train_loss_x = np.append(train_loss_x, epoch)
                    plt.semilogy(train_loss_x, train_loss_y, 'r-', lw=2)
                    plt.xlabel('Epoch')
                    plt.ylabel('Training error')
                    plt.title('IL = 4, FL = 14')
                    plt.grid(True)
                    plt.pause(0.05)

    #plt.savefig('foo.jpg')
    #plt.show()

np.save("loss_mat.npy",loss_mat)
np.savetxt("loss_mat.txt",loss_mat,delimiter=" ")


if save_model:
    print('Saving model......')
    torch.save(mlp, 'mlp_mnist_10epoch_train.pkl')
    # >>>>>>>>Test neural network performance

if load_model:
    print('Loading model......')
    mlp = torch.load('mlp_mnist_10epoch_train.pkl')

# >>>>>>>>>>>>>>>>>>>>>>>>Inference
test_output = mlp(test_x)
pred_y = torch.max(test_output, 1)[1].data.squeeze()
accuracy = sum(pred_y == test_y) / float(test_y.size(0))

print(accuracy, 'prediction accuracy!')
print('End of testing!')
