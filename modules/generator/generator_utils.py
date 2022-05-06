import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class StyleConv(nn.Module):
    def __init__(self, num_input, num_output):
        super(StyleConv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_channels=num_input, out_channels=num_output, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.upsample(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = F.relu(out)
        return out


class UpConv(nn.Module):
    def __init__(self, num_input, num_output):
        super(UpConv, self).__init__()
        self.conv1 = nn.Conv2d(num_input, num_output, kernel_size=1, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, o):
        out = self.conv1(o)
        out = F.relu(out)
        print(out.shape)
        out = self.upsample(out)
        out = F.relu(out)
        return out


class Warp(nn.Module):
    """Batch B -> Images, C -> Channels, H -> Height, W -> Width
             grid is a tensor of batch b operations defining h height and
            w width pixels, and in which xy locations from the input should be sampled."""

    # def warp_image(self, phi, x_enc, m):
    #     b, c, h, w = x_enc.size()
    #     # print(phi)
    #     # mesh grid
    #     xx = torch.arange(0, w).view(1, -1).repeat(h, 1)
    #     yy = torch.arange(0, h).view(-1, 1).repeat(1, w)
    #     xx = xx.view(1, 1, h, w).repeat(b, 1, 1, 1)
    #     yy = yy.view(1, 1, h, w).repeat(b, 1, 1, 1)
    #     grid = torch.cat((xx, yy), 1).float()

    #     if x_enc.is_cuda:
    #         grid = grid.cuda()

    #     vgrid = Variable(grid) + phi

    #     # scale grid to [-1,1]
    #     vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(w - 1, 1) - 1.0
    #     vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(h - 1, 1) - 1.0

    #     # vgrid = vgrid.permute(0, 2, 3, 1)
    #     phi = phi.permute(0, 2, 3, 1)
    #     vgrid = vgrid + 2 * phi
    #     output = F.grid_sample(x_enc, vgrid)
    #     mask = torch.autograd.Variable(torch.ones(x_enc.size()))
    #     mask = F.grid_sample(mask, vgrid)

    #     mask[mask < 0.9999] = 0
    #     mask[mask > 0] = 1

    #     return output * m

    def flow_warp(self, x, flow, padding_mode='zeros'):
        """Warp an image or feature map with optical flow
        Args:
            x (Tensor): size (n, c, h, w)
            flow (Tensor): size (n, 2, h, w), values range from -1 to 1 (relevant to image width or height)
            padding_mode (str): 'zeros' or 'border'

        Returns:
            Tensor: warped image or feature map
        """
        assert x.size()[-2:] == flow.size()[-2:]
        n, _, h, w = x.size()
        x_ = torch.arange(w).view(1, -1).expand(h, -1)
        y_ = torch.arange(h).view(-1, 1).expand(-1, w)
        grid = torch.stack([x_, y_], dim=0).float()
        grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
        grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
        grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
        grid += 2 * flow
        grid = grid.permute(0, 2, 3, 1)
        return F.grid_sample(x, grid)
