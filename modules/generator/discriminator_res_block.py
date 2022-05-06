import torch
from torch import nn
from torch.nn import functional as F

from generator.generator_utils import StyleConv, Warp, UpConv


class DiscriminatorResidualBlock(nn.Module):
    def __init__(self, num_input, num_output, kernel_size):
        self.conv1 = nn.Conv2d(in_channels=num_input, out_channels=num_output, kernel_size=kernel_size, stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_output, out_channels=2, kernel_size=kernel_size, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=num_output, kernel_size=kernel_size, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=2, out_channels=num_output, kernel_size=kernel_size, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_output)
        self.bn2 = nn.BatchNorm2d(2)
        self.bn3 = nn.BatchNorm2d(num_output)
        self.bn4 = nn.BatchNorm2d(num_output)
        self.styleConv = StyleConv(num_input, num_output)
        self.upConv = UpConv(num_input, num_output)

    def forward(self, x_enc, s, u, first_block=False, last_block=False):
        flow_field = []
        style_conv_out = self.styleConv(s)
        style_conv_out = self.conv2(style_conv_out)
        style_conv_out = self.bn2(style_conv_out)
        style_conv_out = F.relu(style_conv_out)

        if not first_block:

            # masked_field = style_conv_out[0]
            # flow_field.append(style_conv_out[1])
            # flow_field.append(style_conv_out[2])

            m = torch.sigmoid(style_conv_out)
            conv_m = self.conv3(m)
            conv_m = self.bn3(conv_m)
            conv_m = F.relu(conv_m)
            phi = torch.tanh(style_conv_out)
            # print(phi.shape)
            # print(x_enc)

            warp = Warp()
            # Added this bcz there was dimension mis-match with m and x_enc
            warped_image = warp.flow_warp(x_enc, phi)
            masked_image = warped_image * conv_m

            to_rgb = masked_image + u
            if last_block:
                return None, to_rgb
            up_conv_output = self.upConv(to_rgb)
            style_conv_out = self.conv4(style_conv_out)
            style_conv_out = self.bn4(style_conv_out)
            style_conv_out = F.relu(style_conv_out)
            style_conv_out = masked_image + style_conv_out
        else:
            style_conv_out = self.conv4(style_conv_out)
            style_conv_out = self.bn4(style_conv_out)
            style_conv_out = F.relu(style_conv_out)
            up_conv_output = self.upConv(style_conv_out)
            style_conv_out = style_conv_out
        print(style_conv_out, up_conv_output)
        return style_conv_out, up_conv_output