import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
#import matplotlib.
#hexin  hexin
from format_input_fixed_point import Formatting
from format_input_fixed_point import apply_format, apply_format_inplace

torch.manual_seed(1)


EPOCH = 0
BATCH_SIZE = 50
LR = 0.01
DOWNLOAD_MNIST = True
IL = torch.IntTensor([4])
FL = torch.IntTensor([12])
config = torch.IntTensor([0])

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



def fixed_point_hook(self,input, output):
    output = apply_format('FXP',output,4,16)


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
        self.formatting = Formatting()
        self.activation = nn.Sigmoid()
        self.out = nn.Linear(100,10)

    def forward(self,x, config):
        x = x.view(x.size(0),-1)
        x = self.hidden1(x)
        if(torch.equal(config, torch.IntTensor([1]))):
            x = self.formatting(x)
        x = self.activation(x)
        output = self.out(x)
        return output #print(x)

mlp=MLP()
#print(mlp)

optimizer = torch.optim.SGD(mlp.parameters(), lr=LR, momentum=0.9) 
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x,y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = mlp(b_x,config)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = mlp(test_x,config)
            pred_y = torch.max(test_output,1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y)/float(test_y.size(0))
            print('Epoch: ',epoch,'| train loss : %.4f' %loss.data[0], '| test accuracy: %.2f' % accuracy)
            


linear1 = list(mlp.hidden1.parameters())
#out = list(mlp.out.parameters())

#print(linear1)   parameter -> contain both weight and bias
#print(linear1[0]) parameter -> contain weight
#print(linear1[0].data)   weight tensor

linear1[0].data=apply_format_inplace('FXP',linear1[0].data,4,12)
#apply_format('FXP',linear1[0],4,12)



#Test neural network performance




config = torch.IntTensor([0])
mlp.hidden1.register_forward_hook(fixed_point_hook)
mlp.activation.register_forward_hook(fixed_point_hook)
mlp.out.register_forward_hook(fixed_point_hook)



test_output = mlp(test_x[:10],config)
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')

print('End of testing!')
