import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
#import matplotlib.
#hexin  hexin
from format_input_fixed_point import Formatting
from format_input_fixed_point import apply_format,apply_format_inplace

torch.manual_seed(1)


EPOCH = 1
BATCH_SIZE = 50
LR = 0.01
DOWNLOAD_MNIST = True
# IL = torch.IntTensor([4])
# FL = torch.IntTensor([12])
IL = 4
FL = 12
config = torch.IntTensor([0])
nonideal_train = 0
nonideal_inference = 0

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
    apply_format_inplace('FXP', output.data, 4, 12)


def fixed_point_back_hook(self, grad_in, grad_out):
    #print('Inside ' + self.__class__.__name__ + ' backward')
    #print(grad_in)
    if grad_in[0] is not None:
        apply_format_inplace('FXP', grad_in[0].data, 4, 12)
    #print('>>>>>>>>>>>>>>>>>>>>>')



'''
def fixed_point_back_hook(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input tuple size: ', grad_input.__len__())
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output tuple size: ', grad_output.__len__())
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size:', grad_input[0].size())
    print('grad_input size:', grad_input[1].size())
    print('grad_input size:', grad_input[2].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].data.norm())
'''


class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1, 
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        '''
        self.hidden1 = nn.Linear(784,100)
        #self.formatting = Formatting()
        self.activation = nn.Sigmoid()
        self.out = nn.Linear(100,10)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.hidden1(x)
        #if(torch.equal(config, torch.IntTensor([1]))):
        #    x = self.formatting(x)
        x = self.activation(x)
        output = self.out(x)
        return output #print(x)

mlp=MLP()
#add backward hooker on each layer
mlp.hidden1.register_backward_hook(fixed_point_back_hook)
mlp.activation.register_backward_hook(fixed_point_back_hook)
mlp.out.register_backward_hook(fixed_point_back_hook)


optimizer = torch.optim.SGD(mlp.parameters(), lr=LR, momentum=0.9) 
loss_func = nn.CrossEntropyLoss()

#>>>>>>>>Dump neural network weight
linear = list(mlp.hidden1.parameters())
out = list(mlp.out.parameters())

for epoch in range(EPOCH):
    for step, (x,y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = mlp(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        apply_format_inplace('FXP', linear[0].data, IL, FL)
        apply_format_inplace('FXP', out[0].data, IL, FL)


        if step % 50 == 0:
            test_output = mlp(test_x)
            pred_y = torch.max(test_output,1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y)/float(test_y.size(0))
            print('Epoch: ',epoch,'| train loss : %.4f' %loss.data[0], '| test accuracy: %.2f' % accuracy)


#>>>>>>>>Test neural network performance





#>>>>>>>>Regulate neural network weight into fixed point counterpart
apply_format_inplace('FXP',linear[0].data,IL,FL) #
apply_format_inplace('FXP',out[0].data,IL,FL)    #
apply_format_inplace('FXP',test_x.data,IL,FL)    # input regulation

#print(type(test_x)) #<class 'torch.autograd.variable.Variable'>
#print(type(test_x.data)) #<class 'torch.FloatTensor'>


#config = torch.IntTensor([0])
#>>>>>>>> Add forward hooks to network layer and regulate its intermeidate output to fixed point counterpart
mlp.hidden1.register_forward_hook(fixed_point_hook)
mlp.activation.register_forward_hook(fixed_point_hook)
mlp.out.register_forward_hook(fixed_point_hook)


#>>>>>>>> Testing!!!
test_output = mlp(test_x)
#test_output = mlp(test_x,config)
pred_y = torch.max(test_output, 1)[1].data.squeeze()
accuracy = sum(pred_y == test_y)/float(test_y.size(0))

print(accuracy, 'prediction accuracy!')
print('End of testing!')
