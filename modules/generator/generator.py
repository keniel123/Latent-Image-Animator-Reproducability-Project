import torch

from generator.generator_res_block import ResidualBlock

"x is z(s->t), x_enc is from encoder"


def Generator():
    x_s_to_d = torch.Tensor()
    x_enc = torch.tensor()

    for i in range(0, 6):
        if i == 0:
            style_conv_output1, up_conv_output1 = (ResidualBlock(4, 3, x_enc, firstBlock=True))
        if i == 1:
            style_conv_output2, up_conv_output2 = (ResidualBlock(8, 3, x_enc, style_conv_output1, up_conv_output1))
        elif i == 2:
            style_conv_output3, up_conv_output3 = (ResidualBlock(16, 3, x_enc, style_conv_output2, up_conv_output2))
        elif i == 3:
            style_conv_output4, up_conv_output4 = (ResidualBlock(32, 3, x_enc, style_conv_output3, up_conv_output3))
        elif i == 4:
            style_conv_output5, up_conv_output5 = (ResidualBlock(64, 3, x_enc, style_conv_output4, up_conv_output4))
        elif i == 5:
            style_conv_output6, up_conv_output6 = (ResidualBlock(128, 3, x_enc, style_conv_output5, up_conv_output5))
        elif i == 6:
            x_s_to_d = ResidualBlock(256, (3, 3), x_enc, style_conv_output6, up_conv_output6, lastBlock=True)[1]

    return x_s_to_d
