import torch

from Generator import ResidualBlock

"x is z(s->t), x_enc is from encoder"


def main():
    x = torch.Tensor
    x_enc = torch.tensor()
    style_conv_output = torch.tensor()
    up_conv_output = torch.tensor()

    for i in range(0, 6):
        if i == 0:
            style_conv_output, up_conv_output = (ResidualBlock(4, 3, x_enc, firstBlock=True))
        if i == 1:
            style_conv_output, up_conv_output = (ResidualBlock(8, 3, x_enc, style_conv_output, up_conv_output))
        elif i == 2:
            style_conv_output, up_conv_output = (ResidualBlock(16, 3, x_enc, style_conv_output, up_conv_output))
        elif i == 3:
            style_conv_output, up_conv_output = (ResidualBlock(32, 3, x_enc, style_conv_output, up_conv_output))
        elif i == 4:
            style_conv_output, up_conv_output = (ResidualBlock(64, 3, x_enc, style_conv_output, up_conv_output))
        elif i == 5:
            style_conv_output, up_conv_output = (ResidualBlock(128, 3, x_enc, style_conv_output, up_conv_output))
        elif i == 6:
            x_s_to_d = ResidualBlock(256, (3, 3), x_enc, style_conv_output, up_conv_output, lastBlock=True)[1]


if __name__ == "__main__":
    main()
