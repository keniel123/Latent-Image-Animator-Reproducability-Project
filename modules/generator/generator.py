import torch

from torch import nn

from generator.generator_res_block import GeneratorResidualBlock


class Generator(nn.Module):
    def __init__(self):
        super.__init__()
        self.layer1 = nn.Sequential(GeneratorResidualBlock(512, 512, kernel_size=3))
        self.layer2 = nn.Sequential(GeneratorResidualBlock(512, 512, kernel_size=3))
        self.layer3 = nn.Sequential(GeneratorResidualBlock(512, 512, kernel_size=3))
        self.layer4 = nn.Sequential(GeneratorResidualBlock(512, 512, kernel_size=3))
        self.layer5 = nn.Sequential(GeneratorResidualBlock(512, 256, kernel_size=3))
        self.layer6 = nn.Sequential(GeneratorResidualBlock(256, 128, kernel_size=3))
        self.layer7 = nn.Sequential(GeneratorResidualBlock(128, 64, kernel_size=3))

    def forward(self, x_enc, z_st_to_d):
        s1, u1 = self.layer1(x_enc, z_st_to_d, firstBlock=True)
        s2, u2 = self.layer2(x_enc, s1, u1)
        s3, u3 = self.layer3(x_enc, s2, u2)
        s4, u4 = self.layer4(x_enc, s3, u3)
        s5, u5 = self.layer5(x_enc, s4, u4)
        s6, u6 = self.layer6(x_enc, s5, u5)
        _, u7 = self.layer7(x_enc, s6, u6, lastBlock=True)
        return u7
