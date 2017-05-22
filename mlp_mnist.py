import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
#import matplotlib.
#hexin  hexin
from format_input_fixed_point import Formatting
from format_input_fixed_point import apply_format_inplace

torch.manual_seed(1)


EPOCH = 5
BATCH_SIZE = 50
LR = 0.01
DOWNLOAD_MNIST = True


IL = 4  # IL = torch.IntTensor([4])
FL = 2 # FL = torch.IntTensor([12])
nonideal_train = 0
nonideal_inference = 1

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


# defining hooks
def fixed_point_hook(self, input, output):
    apply_format_inplace('FXP', output.data, IL, FL)


def fixed_point_back_hook(self, grad_in, grad_out):
    #print('Inside ' + self.__class__.__name__ + ' backward')
    #print(grad_in)
    if grad_in[0] is not None:
        apply_format_inplace('FXP', grad_in[0].data, IL, FL)
    #print('>>>>>>>>>>>>>>>>>>>>>')




class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.hidden1 = nn.Linear(784,100)
        #self.formatting = Formatting()
        self.activation = nn.Sigmoid()
        self.out = nn.Linear(100,10)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.hidden1(x)
        # x = self.formatting(x)
        x = self.activation(x)
        output = self.out(x)
        return output

mlp=MLP()
optimizer = torch.optim.SGD(mlp.parameters(), lr=LR, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

#add backward hooker on each layer, to regulate the gradient to fixed point representation
if(nonideal_train):
    mlp.hidden1.register_backward_hook(fixed_point_back_hook)
    mlp.activation.register_backward_hook(fixed_point_back_hook)
    mlp.out.register_backward_hook(fixed_point_back_hook)
    #add foward hooker on each layer, , to regulate the layer out to fixed point representation
    mlp.hidden1.register_forward_hook(fixed_point_hook)
    mlp.activation.register_forward_hook(fixed_point_hook)
    mlp.out.register_forward_hook(fixed_point_hook)


#>>>>>>>>Dump neural network weight to list
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

        if(nonideal_train):
            apply_format_inplace('FXP', linear[0].data, IL, FL)
            apply_format_inplace('FXP', out[0].data, IL, FL)


        if step % 50 == 0:
            test_output = mlp(test_x)
            pred_y = torch.max(test_output,1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y)/float(test_y.size(0))
            print('Epoch: ',epoch,'| train loss : %.4f' %loss.data[0], '| test accuracy: %.2f' % accuracy)


#>>>>>>>>Test neural network performance


#>>>>>>>>Regulate neural network weight into fixed point counterpart
if(nonideal_inference):
    apply_format_inplace('FXP',linear[0].data,IL,FL) # weight regulation
    apply_format_inplace('FXP',out[0].data,IL,FL)    # bias regulation
    apply_format_inplace('FXP',test_x.data,IL,FL)    # input regulation

#print(type(test_x)) #<class 'torch.autograd.variable.Variable'>
#print(type(test_x.data)) #<class 'torch.FloatTensor'>


#>>>>>>>>add forward hooker on each layer if non-ideal inference
if(nonideal_inference and (not nonideal_train )):
    print('applied!!!')
    mlp.hidden1.register_forward_hook(fixed_point_hook)
    mlp.activation.register_forward_hook(fixed_point_hook)
    mlp.out.register_forward_hook(fixed_point_hook)

#>>>>>>>> Testing!!!
test_output = mlp(test_x)
pred_y = torch.max(test_output, 1)[1].data.squeeze()
accuracy = sum(pred_y == test_y)/float(test_y.size(0))

print(accuracy, 'prediction accuracy!')
print('End of testing!')
