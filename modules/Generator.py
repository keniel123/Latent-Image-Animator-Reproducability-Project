import torch
from torch import nn
from torch.nn import functional as F

from network import Warp, StyleConv, UpConv


class ResidualBlock(nn.Module):

    def __init__(self, num_input, kernel_size, x_enc, style_conv_output, up_conv_output, firstBlock=False, lastBlock=False):
        super().__init__()
        self.conv1 = nn.Conv2d(num_input, num_input*2, kernel_size=kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_input*2, num_input*2, kernel_size=kernel_size, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_input*2)
        self.bn2 = nn.BatchNorm2d(num_input*2)
        self.x_enc = x_enc
        self.first_block = firstBlock
        self.last_block = lastBlock
        self.style_conv_output = style_conv_output
        self.up_conv_output = up_conv_output

    def forward(self, x):
        flow_field = []
        # TODO - Send params to StyleConv zs->t and xi
        style_conv_out = StyleConv(x)
        style_conv_out = self.conv2(style_conv_out)
        style_conv_out = self.bn2(style_conv_out)
        style_conv_out = F.relu(style_conv_out)

        if not self.first_block:
            masked_field = style_conv_out.children()[0]
            flow_field.append(style_conv_out.children()[1])
            flow_field.append(style_conv_out.children()[2])

            m = torch.sigmoid(masked_field)
            phi = torch.tanh(flow_field)

            warped_image = Warp.warp_image(phi, self.x_enc, m)

            warped_image = self.conv2(warped_image)
            warped_image = self.bn2(warped_image)
            warped_image = F.relu(warped_image)

            to_rgb = warped_image + self.generatorout
            if last_block:
                return None, to_rgb
            up_conv_output = UpConv(to_rgb)
            style_conv_out = warped_image + style_conv_out
        else:
            up_conv_output = UpConv(style_conv_out)
            style_conv_out = style_conv_out

        return style_conv_out, up_conv_output
