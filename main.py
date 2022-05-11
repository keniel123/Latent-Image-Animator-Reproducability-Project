import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import transforms
import torchvision
from torchvision.utils import save_image
from modules.discriminator import Discriminator
from modules.encoder import Encoder
from itertools import chain
import torch.nn.functional as F

from modules.evaluation import calculate_lpips_score, calculate_aed_score
from modules.generator import Generator
from modules.lmd import LinearMotionDecomposition
from modules.loss import LossFunctions
from modules.lossf import LossModel
import os
import cv2
import sys
from torch.utils.tensorboard import SummaryWriter

from modules.preprocessing import *
from modules.utils import get_device, save_model, load_model

PATH = "model.pt"


def get_images(folder):
    for image in os.listdir(folder):
        image_list = []
        img = cv2.imread(folder + "/" + image)
        image_list.append(img)
    return image_list


def train(dataset_name, device):
    # build the model
    source_Encoder = Encoder(3, True).to(device)
    driver_Encoder = Encoder(3, False).to(device)

    discriminator = Discriminator().to(device)
    loss = LossModel().to(device)
    lmd = LinearMotionDecomposition().to(device)
    generator = Generator().to(device)
    # define the loss function and the optimiser
    criterion = nn.BCELoss()

    # define optimisers
    generator_optimiser = optim.Adam(
        chain(lmd.parameters(), source_Encoder.parameters(), driver_Encoder.parameters(), generator.parameters()),
        lr=2e-3)
    discriminator_optimiser = optim.Adam(discriminator.parameters(), lr=2e-3)

    # get training set
    training_set = get_dataset(
        TAICHI_TRAINING_IMAGES_VIDEOS_SET_FOLDER if dataset_name == "taichi" else VOXCELEB_TRAINING_IMAGES_VIDEOS_SET_FOLDER)
    writer = SummaryWriter(f'runs/LIA/training/tensorboard')
    step = 0
    # training loop
    if os.path.exists(PATH) and os.path.getsize(PATH) != 0:
        load_model(source_Encoder, driver_Encoder, discriminator, generator, generator_optimiser,
                   discriminator_optimiser, PATH)

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
                writer.add_scalar('Generator loss', generator_loss, global_step=step)
                writer.add_scalar('Discriminator loss', discriminator_loss.item(), global_step=step)

                writer.add_graph(source_Encoder, data)

                # writer.add_embedding(target_latent_code)

                step += 1
                save_image(reconstructed_image[0],
                           GENERATED_TRAINING_DATA_SET_FOLDER + "/{}.jpg".format(count))
                print("save image " + str(count))
                print(generator_losses)
                print(discriminator_losses)

        #             save_image_to_folder(GENERATED_DATA_SET_FOLDER + "/%#05d.jpg" % (3), reconstructed_image[0].numpy())
        #         save_image_to_folder(
        #             GENERATED_DATA_SET_FOLDER + '/%#05d' + '/%#05d' + GENERATED_FRAMES_FOLDER % epoch, i,
        #             reconstructed_image)
        if epoch % 5 == 0:
            save_model(source_Encoder, driver_Encoder, discriminator, generator, generator_optimiser,
                       discriminator_optimiser, PATH)
        return source_Encoder, driver_Encoder, generator, lmd, discriminator


def test(src_encoder, targ_encoder, generator, lmd, discriminator, dataset_name):
    with torch.no_grad():

        src_encoder.eval()
        targ_encoder.eval()
        generator.eval()
        lmd.eval()
        discriminator.eval()

        test_set = get_dataset(
            TAICHI_TESTING_IMAGES_VIDEOS_SET_FOLDER if dataset_name == "taichi" else VOXCELEB_TESTING_IMAGES_VIDEOS_SET_FOLDER)
        # define the loss function and the optimiser
        criterion = nn.BCELoss()
        loss = LossModel()
        generator_losses = []
        discriminator_losses = []
        lpips_losses = []
        aed_losses = []
        writer = SummaryWriter(f'runs/LIA/test/tensorboard')
        step = 0
        for i in range(len(test_set)):
            source_image, driving_frames = test_set[i][0], test_set[i][1:]
            temp_gen_losses = []
            temp_disc_losses = []
            temp_lpips_losses = []
            temp_aed_losses = []
            temp_l1_losses = []
            temp_adv_losses = []
            temp_perceptual_losses = []
            count = 0
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
                temp_l1_losses.append(gen_losses['reconstruction'])
                temp_adv_losses.append(gen_losses['adversarial_loss'])
                temp_perceptual_losses.append(np.sum(gen_losses['perceptual']))
                ##evaluation metrics

                lpips_Score = calculate_lpips_score(source_image, driving_image)
                aed_Score = calculate_aed_score(source_image, driving_image)
                temp_lpips_losses.append(lpips_Score)
                temp_aed_losses.append(aed_Score)

                #### save test image
                save_images_to_folder(reconstructed_image, dataset_name, "video " + str(i), count)
                count += 1

            writer.add_scalar(dataset_name + '  Generator loss ', np.mean(temp_gen_losses), global_step=step)
            writer.add_scalar(dataset_name + ' Discriminator loss', np.mean(temp_disc_losses), global_step=step)
            writer.add_scalar(dataset_name + ' AED Loss', np.mean(temp_aed_losses), global_step=step)
            writer.add_scalar(dataset_name + ' Lpips Loss', np.mean(temp_lpips_losses), global_step=step)
            writer.add_scalar(dataset_name + ' L1 Loss', np.mean(temp_l1_losses), global_step=step)
            writer.add_scalar(dataset_name + ' Adversarial Loss', np.mean(temp_adv_losses), global_step=step)
            writer.add_scalar(dataset_name + ' Perceptual Loss', np.mean(temp_perceptual_losses), global_step=step)
            step += 1
            generator_losses.append(temp_gen_losses)
            discriminator_losses.append(temp_disc_losses)
            aed_losses.append(temp_aed_losses)
            lpips_losses.append(temp_lpips_losses)

            generate_reference_image(src_encoder, generator, source_image, driving_frames[len(driving_frames) // 2],
                                     count)

        writer.add_scalar(dataset_name + ' Total Mean Generator loss ', np.mean(np.hstack(temp_gen_losses)))
        writer.add_scalar(dataset_name + ' Total Mean Discriminator loss', np.mean(np.hstack(temp_disc_losses)))
        writer.add_scalar(dataset_name + ' Total Mean AED Loss', np.mean(np.hstack(temp_aed_losses)))
        writer.add_scalar(dataset_name + ' Total Mean Lpips Loss', np.mean(np.hstack(temp_lpips_losses)))
        writer.add_scalar(dataset_name + ' Total Mean L1 Loss', np.mean(np.hstack(temp_l1_losses)))
        writer.add_scalar(dataset_name + ' Total Mean Adversarial Loss', np.mean(np.hstack(temp_adv_losses)))
        writer.add_scalar(dataset_name + ' Total Mean Perceptual Loss', np.mean(np.hstack(temp_perceptual_losses)))


def generate_reference_image(encoder, generator, source_image, dataset_name, count):
    source_features, source_latent_code = encoder(source_image)
    reference_image = generator(source_features, source_latent_code)
    save_images_to_folder(reference_image, dataset_name, "references", count)


def main():
    # training taichi
    src_encoder, targ_encoder, generator, lmd, discriminator = train("taichi", get_device())
    test(src_encoder, targ_encoder, generator, lmd, discriminator, "taichi")

    # training voxceleb
    src_encoder, targ_encoder, generator, lmd, discriminator = train("voxceleb", get_device())
    test(src_encoder, targ_encoder, generator, lmd, discriminator, "voxceleb")


if __name__ == "__main__":
    main()
