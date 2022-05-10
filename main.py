import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.utils import save_image
from modules.discriminator import Discriminator
from modules.encoder import Encoder
from itertools import chain
import torch.nn.functional as F

from modules.generator import Generator
from modules.lmd import LinearMotionDecomposition
from modules.loss import LossFunctions
from modules.lossf import LossModel
import os
import cv2
import sys

from modules.preprocessing import *

def get_images(folder):
    for image in os.listdir(folder):
        image_list = []
        img = cv2.imread(folder + "/" + image)
        image_list.append(img)
    return image_list


def train(dataset_name):
    # build the model
    source_Encoder = Encoder(3, True)
    driver_Encoder = Encoder(3, False)

    discriminator = Discriminator()
    loss = LossModel()
    lmd = LinearMotionDecomposition()
    generator = Generator()

    # define the loss function and the optimiser
    criterion = nn.BCELoss()

    # define optimisers
    generator_optimiser = optim.Adam(
        chain(lmd.parameters(), source_Encoder.parameters(), driver_Encoder.parameters(), generator.parameters()),
        lr=2e-3)
    discriminator_optimiser = optim.Adam(discriminator.parameters(), lr=2e-3)

    # get training set
    training_set = get_dataset(TAICHI_TRAINING_IMAGES_VIDEOS_SET_FOLDER if dataset_name == "taichi" else VOXCELEB_TRAINING_IMAGES_VIDEOS_SET_FOLDER )

    # training loop
    for epoch in range(1):
        generator_losses = []
        discriminator_losses = []
        for i in range(len(training_set)):
            source_image, driving_frames = training_set[i][0], training_set[i][1:]
            dataloader = get_dataloader(driving_frames, 4)
            count = 0
            for data in dataloader:
                count += 1

                # driving_image = driving_image.unsqueeze(0)
                generator_optimiser.zero_grad()
                discriminator_optimiser.zero_grad()
                source_features, source_latent_code = source_Encoder(source_image.unsqueeze(0))

                motion_magnitudes = driver_Encoder(data)
                # print(motion_magnitudes.shape)
                target_latent_code = lmd.generate_target_code(source_latent_code, motion_magnitudes)

                reconstructed_image = generator(source_features, target_latent_code)
                discriminator_real = discriminator(data).reshape(-1)
                discriminator_fake = discriminator(reconstructed_image).reshape(-1)

                discriminator_loss_real = criterion(discriminator_real, torch.ones_like(discriminator_real))
                discriminator_loss_fake = criterion(discriminator_fake, torch.ones_like(discriminator_fake))
                discriminator_loss = (discriminator_loss_real + discriminator_loss_fake / 2)
                discriminator_loss.backward(retain_graph=True)
                discriminator_optimiser.step()

                gen_losses = loss(reconstructed_image, data, discriminator_fake)
                loss_values = [val.mean() for val in gen_losses.values()]
                generator_loss = sum(loss_values)
                generator_loss.backward()
                generator_optimiser.step()

                # keep track of the loss and update the stats
                generator_losses.append(generator_loss.item())
                discriminator_losses.append(discriminator_loss.item())

                # save_image_to_folder(
                #     GENERATED_DATA_SET_FOLDER + "/{}/{}".format(epoch, i) + GENERATED_FRAMES_FOLDER
                #     , epoch, i,
                #     reconstructed_image.detach().numpy())

                save_image(reconstructed_image[0],
                           GENERATED_DATA_SET_FOLDER + "/{}.jpg".format(count))
                print("save image " + str(count))
                print(generator_losses)
                print(discriminator_losses)

        #             save_image_to_folder(GENERATED_DATA_SET_FOLDER + "/%#05d.jpg" % (3), reconstructed_image[0].numpy())
        #         save_image_to_folder(
        #             GENERATED_DATA_SET_FOLDER + '/%#05d' + '/%#05d' + GENERATED_FRAMES_FOLDER % epoch, i,
        #             reconstructed_image)
        source_Encoder.eval()
        driver_Encoder.eval()
        lmd.eval()
        discriminator.eval()

        test(source_Encoder, driver_Encoder, generator, lmd, discriminator, dataset_name)


def test(src_encoder, targ_encoder, generator, lmd,discriminator, dataset_name):
    test_set = get_dataset(TAICHI_TESTING_IMAGES_VIDEOS_SET_FOLDER if dataset_name == "taichi" else VOXCELEB_TESTING_IMAGES_VIDEOS_SET_FOLDER)
    # define the loss function and the optimiser
    criterion = nn.BCELoss()
    loss = LossModel()
    generator_losses = []
    discriminator_losses = []
    for i in range(len(test_set)):
        source_image, driving_frames = test_set[i][0], test_set[i][1:]
        temp_gen_losses = []
        temp_disc_losses = []
        for driving_image in driving_frames:
            source_features, source_latent_code = src_encoder(source_image)

            motion_magnitudes = targ_encoder(driving_image)

            target_latent_code = lmd.generate_target_code(source_latent_code, motion_magnitudes)

            reconstructed_image = generator(source_features, target_latent_code)

            discriminator_real = discriminator(driving_image.unsqueeze(0)).reshape(-1)
            discriminator_fake = discriminator(reconstructed_image).reshape(-1)

            discriminator_loss_real = criterion(discriminator_real, torch.ones_like(discriminator_real))
            discriminator_loss_fake = criterion(discriminator_fake, torch.ones_like(discriminator_fake))
            discriminator_loss = (discriminator_loss_real + discriminator_loss_fake / 2)

            gen_losses = loss(reconstructed_image, driving_image.unsqueeze(0), discriminator_fake)
            loss_values = [val.mean() for val in gen_losses.values()]
            generator_loss = sum(loss_values)

            temp_gen_losses.append(generator_loss.item())
            temp_disc_losses.append(discriminator_loss.item())
        generator_losses.append(temp_gen_losses)
        discriminator_losses.append(temp_disc_losses)


def main():
    # build the model
    source_Encoder = Encoder(3, True)
    driver_Encoder = Encoder(3, False)

    discriminator = Discriminator()
    loss = LossModel()
    lmd = LinearMotionDecomposition()
    generator = Generator()

    # define the loss function and the optimiser
    criterion = nn.BCELoss()

    # define optimisers
    generator_optimiser = optim.Adam(
        chain(lmd.parameters(), source_Encoder.parameters(), driver_Encoder.parameters(), generator.parameters()),
        lr=2e-3)
    discriminator_optimiser = optim.Adam(discriminator.parameters(), lr=2e-3)

    # get training set
    training_set = get_dataset(TAICHI_TRAINING_IMAGES_VIDEOS_SET_FOLDER)

    # training loop
    for epoch in range(1):
        generator_losses = []
        discriminator_losses = []
        for i in range(len(training_set)):
            source_image, driving_frames = training_set[i][0], training_set[i][1:]
            dataloader = get_dataloader(driving_frames, 4)
            count = 0
            for data in dataloader:
                count += 1

                # driving_image = driving_image.unsqueeze(0)
                generator_optimiser.zero_grad()
                discriminator_optimiser.zero_grad()
                source_features, source_latent_code = source_Encoder(source_image.unsqueeze(0))

                motion_magnitudes = driver_Encoder(data)
                # print(motion_magnitudes.shape)
                target_latent_code = lmd.generate_target_code(source_latent_code, motion_magnitudes)

                reconstructed_image = generator(source_features, target_latent_code)
                discriminator_real = discriminator(data).reshape(-1)
                discriminator_fake = discriminator(reconstructed_image).reshape(-1)

                discriminator_loss_real = criterion(discriminator_real, torch.ones_like(discriminator_real))
                discriminator_loss_fake = criterion(discriminator_fake, torch.ones_like(discriminator_fake))
                discriminator_loss = (discriminator_loss_real + discriminator_loss_fake / 2)
                discriminator_loss.backward(retain_graph=True)
                discriminator_optimiser.step()

                gen_losses = loss(reconstructed_image, data, discriminator_fake)
                loss_values = [val.mean() for val in gen_losses.values()]
                generator_loss = sum(loss_values)
                generator_loss.backward()
                generator_optimiser.step()

                # keep track of the loss and update the stats
                generator_losses.append(generator_loss.item())
                discriminator_losses.append(discriminator_loss.item())

                # save_image_to_folder(
                #     GENERATED_DATA_SET_FOLDER + "/{}/{}".format(epoch, i) + GENERATED_FRAMES_FOLDER
                #     , epoch, i,
                #     reconstructed_image.detach().numpy())

                save_image(reconstructed_image[0],
                           GENERATED_DATA_SET_FOLDER + "/{}.jpg".format(count))
                print("save image " + str(count))
                print(generator_losses)
                print(discriminator_losses)

        #             save_image_to_folder(GENERATED_DATA_SET_FOLDER + "/%#05d.jpg" % (3), reconstructed_image[0].numpy())
        #         save_image_to_folder(
        #             GENERATED_DATA_SET_FOLDER + '/%#05d' + '/%#05d' + GENERATED_FRAMES_FOLDER % epoch, i,
        #             reconstructed_image)
        source_Encoder.eval()
        driver_Encoder.eval()
        lmd.eval()
        discriminator.eval()




if __name__ == "__main__":
    main()
