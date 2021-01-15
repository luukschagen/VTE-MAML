import torch
from torch import nn
from torch import distributions as dist

class Convnet(nn.Module):
    def __init__(self, input_size, conv_layers, output_size):
        super(Convnet, self).__init__()
        final_conv_size = input_size[1]
        for layer in conv_layers:
            final_conv_size = int(((final_conv_size+2*1-1*2-1)/2+1))

        self.final_conv_size = [conv_layers[-1], final_conv_size, final_conv_size]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(input_size[0], conv_layers[0], 3, 2, 1, 1))
        self.layers.append(nn.BatchNorm2d(conv_layers[0]))
        self.layers.append(nn.ReLU())
        for layer in range(1, len(conv_layers)):
            self.layers.append(nn.Conv2d(conv_layers[layer-1], conv_layers[layer], 3, 2, 1, 1))
            self.layers.append(nn.BatchNorm2d(conv_layers[layer]))
            self.layers.append(nn.ReLU())
        self.final_layer = nn.Linear(self.final_conv_size[0], output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = x.mean(dim=(0, 2, 3)).view(1,-1)
        x = self.final_layer(x)
        return x


class Sequential(nn.Module):

    def __init__(self, modulelist):
        super(Sequential, self).__init__()
        self.modulelist = modulelist

    def forward(self, x):
        for module in self.modulelist:
            x = module(x)
        return x


def log_likelihood(outputs, targets, reduction='sum'):
    if reduction != 'sum':
        raise NotImplementedError
    p_x = dist.Normal(outputs[:, 0].view(-1, 1), torch.exp(outputs[:, 1]).view(-1, 1))
    return -1 * p_x.log_prob(targets).sum()