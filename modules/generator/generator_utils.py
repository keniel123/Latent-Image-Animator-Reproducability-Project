import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class StyleConv(nn.Module):
    def __init__(self, num_input, num_output):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.conv1 = nn.Conv2d(num_input, num_output, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.upsample(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = F.relu(out)
        return out


class UpConv(nn.Module):
    def __init__(self, num_input, num_output):
        super().__init__()
        self.conv1 = nn.Conv2d(num_input, num_output, kernel_size=1, stride=1, padding=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

    def forward(self, o):
        out = self.conv1(o)
        out = F.relu(out)
        out = self.upsample(out)
        out = F.relu(out)
        return out


class Warp(nn.Module):
    """Batch B -> Images, C -> Channels, H -> Height, W -> Width
             grid is a tensor of batch b operations defining h height and
            w width pixels, and in which xy locations from the input should be sampled."""

    def warp_image(self, phi, x_enc, m):
        b, c, h, w = x_enc.size()
        # mesh grid
        xx = torch.arange(0, w).view(1, -1).repeat(h, 1)
        yy = torch.arange(0, h).view(-1, 1).repeat(1, w)
        xx = xx.view(1, 1, h, w).repeat(b, 1, 1, 1)
        yy = yy.view(1, 1, h, w).repeat(b, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x_enc.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + phi

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(w - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(h - 1, 1) - 1.0

        # vgrid = vgrid.permute(0, 2, 3, 1)
        phi = phi.permute(0, 2, 3, 1)
        vgrid = vgrid + 2 * phi
        output = F.grid_sample(x_enc, vgrid)
        # mask = torch.autograd.Variable(torch.ones(xenc.size())).cuda()
        # mask = F.grid_sample(mask, vgrid)
        #
        # mask[mask < 0.9999] = 0
        # mask[mask > 0] = 1

        return output * m


class Residual(nn.Module):
    """The Residual block of ResNet."""

    def __init__(self, input_channels, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        """BatchNorm2d is the number of dimensions/channels
         that output from the last layer and come in to the batch norm layer"""
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        return out
