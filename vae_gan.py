

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

import vae as vae
import reporter as rp
import visualization as viz
import annealer

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.disc(x)



# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
vae_lr = 9e-6
gen_lr = 1e-4
dis_lr = 9e-5
z_dim = 266
image_dim = 28 * 28 * 1  # 784
batch_size = 128
num_epochs = 50
negative_slope = 0.01
skip_iteration = 2


# autoencoder = VAE(z_dim, 512).to(device)
encoder_layers = nn.Sequential()
encoder_layers.add_module('fc1', nn.Linear(image_dim, 512))
encoder_layers.add_module('relu1', nn.LeakyReLU(negative_slope))
encoder_layers.add_module('fc2', nn.Linear(512, 256))
encoder_layers.add_module('relu2', nn.LeakyReLU(negative_slope))
encoder_layers.add_module('fc3', nn.Linear(256, z_dim))
encoder_layers.add_module('relu3', nn.LeakyReLU(negative_slope))

encoder = vae.Encoder(encoder_layers)


generator_layers = nn.Sequential()
generator_layers.add_module('fc1', nn.Linear(z_dim, 256))
generator_layers.add_module('relu1', nn.LeakyReLU(negative_slope))
generator_layers.add_module('fc2', nn.Linear(256, 512))
generator_layers.add_module('relu2', nn.LeakyReLU(negative_slope))
generator_layers.add_module('fc3', nn.Linear(512, image_dim))
generator_layers.add_module('relu3', nn.LeakyReLU(negative_slope))

generator = vae.Decoder(generator_layers)

discriminator = Discriminator(image_dim)


dataset_mnist = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
data_loader_mnist = torch.utils.data.DataLoader(dataset_mnist, batch_size=batch_size, shuffle=True)


discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=dis_lr)
generator_optimizer = optim.Adam(generator.parameters(), lr=gen_lr)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=vae_lr)

kl_annealer = annealer.Annealer(0, 1, 0.1)

num_mini_batches = len(data_loader_mnist)
epoch_mse_loss = 1.0
epoch_kl_loss = 1.0
epoch_discriminator_loss_real = 1.0
epoch_discriminator_loss_fake = 1.0
epoch_discriminator_loss = 1.0
epoch_generator_loss = 1.0

reporter = rp.Reporter('./logs/vae_gan', './logs/snapshots')

flip = False
for epoch in range(num_epochs):
    if epoch % skip_iteration == 0:
        flip = not flip

    print(f"Epoch [{epoch}/{num_epochs}]")
    if(epoch > 0):
        reporter.add_scalar('MSE Loss', epoch_mse_loss, epoch)
        reporter.add_scalar('Discriminator Loss', epoch_discriminator_loss, epoch)
        reporter.add_scalar('Discriminator Loss Real', epoch_discriminator_loss_real, epoch)
        reporter.add_scalar('Discriminator Loss Fake', epoch_discriminator_loss_fake, epoch)
        reporter.add_scalar('Generator Loss', epoch_generator_loss, epoch)


    print(f"Epoch MSE Loss: {epoch_mse_loss}")
    epoch_mse_loss = 0
    epoch_kl_loss = 0
    epoch_discriminator_loss = 0
    epoch_discriminator_loss_real = 0
    epoch_discriminator_loss_fake = 0
    epoch_generator_loss = 0

    for i, (data, label) in enumerate (data_loader_mnist):
        data = data.to(device)
        batch_size = data.shape[0]
        # Train discriminator with real data
        real_image = data.view(batch_size, -1)

        discriminator.zero_grad()
        #Expect real image to be classified as 1
        real_image_label = torch.ones(batch_size, 1).to(device)
        output_real = discriminator(real_image)


        real_data_error = nn.BCELoss()(output_real, real_image_label)
        epoch_discriminator_loss_real += real_data_error
        real_data_error.backward(retain_graph=True)

        # Train discriminator with fake data
        z = encoder(real_image)

        fake_image = generator(z)
        #Expect fake image to be classified as 0
        fake_image_label = torch.zeros(batch_size, 1).to(device)
        output_fake = discriminator(fake_image)

        fake_data_error = nn.BCELoss()(output_fake, fake_image_label)
        epoch_discriminator_loss_fake += fake_data_error
        fake_data_error.backward(retain_graph=True)

        discriminator_error = real_data_error + fake_data_error
        epoch_discriminator_loss += discriminator_error
        if flip:
            discriminator_optimizer.step()

        # Train VAE on MSE and KL divergence
        generator.zero_grad()
        encoder.zero_grad()

        z2 = encoder(real_image)
        x_hat = generator(z2)

        shaped_real_image = real_image.view(batch_size, 1, 28, 28)
        MSE = F.mse_loss(x_hat, shaped_real_image, reduction='mean')
        epoch_mse_loss += MSE
        vae_loss = MSE + (kl_annealer(epoch) * encoder.kl)
        vae_loss.backward(retain_graph=True)
        encoder_optimizer.step()
        generator_optimizer.step()

        # Train generator/decoder on discriminator output
        x_hat2 = generator(encoder(real_image))
        generator.zero_grad() 
        discriminator_guess = discriminator(x_hat2)
        generator_error = nn.BCELoss()(discriminator_guess, real_image_label)
        epoch_generator_loss += generator_error
        generator_error.backward()

    if not flip:
            generator_optimizer.step()

    epoch_mse_loss /= num_mini_batches
    epoch_discriminator_loss /= num_mini_batches
    epoch_discriminator_loss_real /= num_mini_batches
    epoch_discriminator_loss_fake /= num_mini_batches
    epoch_generator_loss /= num_mini_batches

viz.plot_latent_encoder(encoder, data_loader_mnist)
viz.plot_generated(generator, z_dim, 20)