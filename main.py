import torch
from torch import nn
from torch import optim
from torchvision import transforms

from modules.discriminator import Discriminator
from modules.encoder import Encoder
from itertools import chain
import torch.nn.functional as F

from modules.generator import Generator
from modules.lmd import LinearMotionDecomposition
from modules.loss import LossFunctions
import os
import cv2
import sys

from modules.preprocessing import get_training_set, TRAINING_IMAGES_VIDEOS_SET_FOLDER, save_image_to_folder, \
    GENERATED_DATA_SET_FOLDER, GENERATED_FRAMES_FOLDER


def get_images(folder):
    for image in os.listdir(folder):
        image_list = []
        img = cv2.imread(folder + "/" + image)
        image_list.append(img)
    return image_list


def main():
    # build the model
    source_Encoder = Encoder(3, True)
    driver_Encoder = Encoder(3, False)

    convert_tensor = transforms.ToTensor()
    discriminator = Discriminator()
    loss = LossFunctions()
    lmd = LinearMotionDecomposition()
    generator = Generator()

    # define the loss function and the optimiser
    criterion = nn.BCELoss()

    # define optimisers
    generator_optimiser = optim.Adam(chain(source_Encoder.parameters(), driver_Encoder.parameters()), lr=2e-3)
    discriminator_optimiser = optim.Adam(discriminator.parameters(), lr=2e-3)

    # get training set
    training_set = get_training_set(TRAINING_IMAGES_VIDEOS_SET_FOLDER)

    # training loop
    for epoch in range(10):
        generator_losses = []
        discriminator_losses = []
        for i in range(len(training_set)):
            source_image, driving_frames = training_set[i][0], training_set[i][1:]
            for driving_image in driving_frames:
                generator_optimiser.zero_grad()
                discriminator_optimiser.zero_grad()
                #print(driving_image)
                #source_features, source_latent_code = source_Encoder(source_image)

                motion_magnitudes = driver_Encoder(driving_image)
                print(motion_magnitudes)
                sys.exit()
                break
        #         target_latent_code = lmd.generate_target_code(source_latent_code, motion_magnitudes)
        #         reconstructed_image = generator(source_features, target_latent_code)
        #
        #         discriminator_loss_real = criterion(driving_image, torch.ones_like(driving_image))
        #         discriminator_loss_fake = criterion(reconstructed_image, torch.ones_like(reconstructed_image))
        #         discriminator_loss = (discriminator_loss_real + discriminator_loss_fake / 2)
        #         discriminator_loss.backward()
        #         discriminator_optimiser.step()
        #
        #         generator_loss = loss.loss_function(reconstructed_image, driving_image)
        #         generator_loss.backward()
        #         generator_optimiser.step()
        #
        #         # loss_values = [val.mean() for val in losses.values()]
        #         # loss = sum(loss_values)
        #
        #         # keep track of the loss and update the stats
        #         generator_losses.append(generator_loss.item())
        #         discriminator_losses.append(discriminator_loss.item())
        #         save_image_to_folder(
        #             GENERATED_DATA_SET_FOLDER + '/%#05d' + '/%#05d' + GENERATED_FRAMES_FOLDER % epoch, i,
        #             reconstructed_image)
        # source_Encoder.eval()
        # driver_Encoder.eval()
        # lmd.eval()
        # discriminator.eval()
        break


if __name__ == "__main__":
    main()


