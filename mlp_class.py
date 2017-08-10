import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.hidden1 = nn.Linear(784,1000)
        #self.formatting = Formatting()
        self.activation = nn.Sigmoid()
        self.hidden2 = nn.Linear(1000, 10)
        self.activation2 = nn.Sigmoid()
        #self.activation = nn.Sigmoid()
        self.out = nn.Linear(1000,10)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.hidden1(x)
        # x = self.formatting(x)
        x = self.activation(x)
        x = self.hidden2(x)
        output = self.activation2(x)
        #output = self.out(x)
        return output