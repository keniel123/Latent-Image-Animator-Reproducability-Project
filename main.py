import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.utils import save_image
from modules.discriminator import Discriminator
from modules.encoder import Encoder
from itertools import chain
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from modules.generator import Generator
from modules.lmd import LinearMotionDecomposition
from modules.loss import LossFunctions
from modules.lossf import LossModel
import os
import cv2
import sys
import yaml
import os.path

from modules.preprocessing import get_training_set, TRAINING_IMAGES_VIDEOS_SET_FOLDER, save_image_to_folder, \
    GENERATED_DATA_SET_FOLDER, GENERATED_FRAMES_FOLDER


def get_images(folder):
    for image in os.listdir(folder):
        image_list = []
        img = cv2.imread(folder + "/" + image)
        image_list.append(img)
    return image_list


def main():
    PATH = "model.pt"

    # Create a SummaryWriter instance
    # SummaryWriter writes event files to log_dir
    log_dir = "logs"
    writer = SummaryWriter(log_dir)
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
    generator_optimiser = optim.Adam(chain(source_Encoder.parameters(), driver_Encoder.parameters()), lr=2e-3)
    discriminator_optimiser = optim.Adam(discriminator.parameters(), lr=2e-3)

    # get training set
    training_set = get_training_set(TRAINING_IMAGES_VIDEOS_SET_FOLDER)

    # training loop
    for epoch in range(10):
        generator_losses = []
        discriminator_losses = []

        # Load the existing model
        if os.path.exists(PATH) and os.path.getsize(PATH) == 0:
            checkpoint = torch.load(PATH)
            epoch = checkpoint['epoch']
            source_Encoder = checkpoint['sourceencoder']
            driver_Encoder = checkpoint['driver_encoder']
            discriminator = checkpoint['discriminator']
            generator = checkpoint['generator']
            generator_optimiser.load_state_dict(checkpoint['generator_optimizer_state_dict'])
            discriminator_optimiser.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
            generator_losses = checkpoint['generator_loss']
            discriminator_losses = checkpoint['discriminator_loss']

        for i in range(len(training_set)):
            source_image, driving_frames = training_set[i][0], training_set[i][1:]
            for driving_image in driving_frames:
                # driving_image = driving_image.unsqueeze(0)
                generator_optimiser.zero_grad()
                discriminator_optimiser.zero_grad()
                source_features, source_latent_code = source_Encoder(source_image)

                motion_magnitudes = driver_Encoder(driving_image)

                target_latent_code = lmd.generate_target_code(source_latent_code, motion_magnitudes)

                reconstructed_image = generator(source_features, target_latent_code)
                discriminator_real = discriminator(driving_image.unsqueeze(0)).reshape(-1)
                discriminator_fake = discriminator(reconstructed_image).reshape(-1)

                discriminator_loss_real = criterion(discriminator_real, torch.ones_like(discriminator_real))
                discriminator_loss_fake = criterion(discriminator_fake, torch.ones_like(discriminator_fake))
                discriminator_loss = (discriminator_loss_real + discriminator_loss_fake / 2)
                discriminator_loss.backward(retain_graph=True)
                discriminator_optimiser.step()

                gen_losses = loss(reconstructed_image, driving_image.unsqueeze(0), discriminator_fake)
                loss_values = [val.mean() for val in gen_losses.values()]
                generator_loss = sum(loss_values)
                generator_loss.backward()
                generator_optimiser.step()

                # keep track of the loss and update the stats
                generator_losses.append(generator_loss.item())
                discriminator_losses.append(discriminator_loss.item())

                torch.save({
                    'epoch': epoch,
                    'sourceencoder': source_Encoder,
                    'driver_encoder': driver_Encoder,
                    'discriminator': discriminator,
                    'generator': generator,
                    'generator_optimizer_state_dict': generator_optimiser.load_state_dict(),
                    'discriminator_optimizer_state_dict': discriminator_optimiser.load_state_dict(),
                    'generator_loss': generator_losses,
                    'discriminator_loss': discriminator_losses,
                }, PATH)

                # Tensorboard
                writer.add_scalar('Generator Loss', generator_losses, epoch)
                writer.add_scalar('Discriminator Loss', discriminator_losses, epoch)

                # save_image_to_folder(
                #     GENERATED_DATA_SET_FOLDER + "/{}/{}".format(epoch, i) + GENERATED_FRAMES_FOLDER
                #     , epoch, i,
                #     reconstructed_image.detach().numpy())

                save_image(reconstructed_image[0],
                           GENERATED_DATA_SET_FOLDER + "/{}.jpg".format(i))

        #             save_image_to_folder(GENERATED_DATA_SET_FOLDER + "/%#05d.jpg" % (3), reconstructed_image[0].numpy())
        #         save_image_to_folder(
        #             GENERATED_DATA_SET_FOLDER + '/%#05d' + '/%#05d' + GENERATED_FRAMES_FOLDER % epoch, i,
        #             reconstructed_image)

        source_Encoder.eval()
        driver_Encoder.eval()
        lmd.eval()
        discriminator.eval()

    # closing the writer for torchvision
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
