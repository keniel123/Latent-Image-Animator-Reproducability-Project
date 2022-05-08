import torch
from torch import nn

import torch
import torchvision
import torch.nn.functional as F

from modules.utils import ImagePyramid


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg19(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg19(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg19(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg19(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


class LossFunctions:
    lambda_perceptual_loss = 10

    def __init__(self):
        super(LossFunctions, self).__init__()

    def reconstruction_loss(self, reconstructed_image, target_image):
        loss = nn.L1Loss(reduction='mean')
        #return torch.sqrt(torch.mean(torch.abs(reconstructed_image - target_image).pow(2)))
        return loss(reconstructed_image, target_image)

    def perceptual_loss(self, reconstructed_image, target_image):
        vgg19Loss = VGGPerceptualLoss()
        return vgg19Loss(reconstructed_image, target_image)

    def adversarial_loss(self, prediction):
        loss = F.softplus(-prediction).mean()
        return loss

    def loss_function(self, reconstructed_image, target_image):
        return self.reconstruction_loss(reconstructed_image, target_image) + \
               (self.lambda_perceptual_loss * self.perceptual_loss(reconstructed_image,
                                                                   target_image)) + self.adversarial_loss(
            reconstructed_image)
