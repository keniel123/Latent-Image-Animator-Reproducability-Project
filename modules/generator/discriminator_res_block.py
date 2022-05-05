import torch
from torch import nn
from torch.nn import functional as F

from generator.generator_utils import StyleConv, Warp, UpConv


class DiscriminatorResidualBlock(nn.Module):
    def __init__(self, num_input, num_output, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=num_input, out_channels=num_output, kernel_size=kernel_size, stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_output, out_channels=num_output, kernel_size=kernel_size, stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(num_input)
        self.bn2 = nn.BatchNorm2d(num_input)

    def forward(self, s, u, first_block=False, last_block=False):
        flow_field = []
        style_conv_out = StyleConv(s)
        style_conv_out = self.conv2(style_conv_out)
        style_conv_out = self.bn2(style_conv_out)
        style_conv_out = F.relu(style_conv_out)

        if not first_block:
            masked_field = style_conv_out.children()[0]
            flow_field.append(style_conv_out.children()[1])
            flow_field.append(style_conv_out.children()[2])

            m = torch.sigmoid(masked_field)
            phi = torch.tanh(flow_field)

            warped_image = Warp.warp_image(phi, s, m)

            warped_image = self.conv2(warped_image)
            warped_image = self.bn2(warped_image)
            warped_image = F.relu(warped_image)

            to_rgb = warped_image + u
            if last_block:
                return None, to_rgb
            up_conv_output = UpConv(to_rgb)
            style_conv_out = warped_image + style_conv_out
        else:
            up_conv_output = UpConv(style_conv_out)
            style_conv_out = style_conv_out

        return style_conv_out, up_conv_output
