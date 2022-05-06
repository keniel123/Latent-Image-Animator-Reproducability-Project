from torch import nn

from generator.discriminator_res_block import DiscriminatorResidualBlock


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(DiscriminatorResidualBlock(512, 512, kernel_size=3))
        self.layer2 = nn.Sequential(DiscriminatorResidualBlock(512, 512, kernel_size=3))
        self.layer3 = nn.Sequential(DiscriminatorResidualBlock(512, 512, kernel_size=3))
        self.layer4 = nn.Sequential(DiscriminatorResidualBlock(512, 512, kernel_size=3))
        self.layer5 = nn.Sequential(DiscriminatorResidualBlock(512, 256, kernel_size=3))
        self.layer6 = nn.Sequential(DiscriminatorResidualBlock(256, 128, kernel_size=3))
        self.layer7 = nn.Sequential(DiscriminatorResidualBlock(128, 64, kernel_size=3))

    def forward(self, x_enc):
        s1, u1 = self.layer1(x_enc)
        s2, u2 = self.layer2(x_enc, s1, u1, first_block=True)
        s3, u3 = self.layer3(x_enc, s2, u2)
        s4, u4 = self.layer4(x_enc, s3, u3)
        s5, u5 = self.layer5(x_enc, s4, u4)
        s6, u6 = self.layer6(x_enc, s5, u5)
        _, u7 = self.layer7(x_enc, s6, u6, last_block=True)
        return u7
