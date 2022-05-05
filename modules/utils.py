from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pool_stride=2):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=pool_stride, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=1, stride=pool_stride, padding=0)  ##used to downsample in resblock

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool1(out)
        residual = self.pool2(residual)
        out += residual
        out = F.relu(out)
        return out


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, magnitudes):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc3 = nn.Linear(input_size, hidden_size)
        self.fc4 = nn.Linear(input_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, magnitudes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        out = F.relu(out)
        out = self.fc5(out)
        return out
