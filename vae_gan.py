

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
import discriminator_tests as dt
import torch

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
skip_iteration = 1


# autoencoder = VAE(z_dim, 512).to(device)
autoencoder = vae.VAE([image_dim, 512, 368, z_dim], [z_dim, 368, 512, image_dim])
ncoder_layers = nn.Sequential()

discriminator = Discriminator(image_dim)


dataset_mnist = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
data_loader_mnist = torch.utils.data.DataLoader(dataset_mnist, batch_size=batch_size, shuffle=True)


discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=dis_lr)
vae_optimizer = optim.Adam(autoencoder.parameters(), lr=gen_lr)

kl_annealer = annealer.Annealer(0, 0.5, 0.03, start_epoch=10)
gan_annealer = annealer.Annealer(0, 1.0, 0.10, start_epoch=10)

num_mini_batches = len(data_loader_mnist)
epoch_mse_loss = 0.0
epoch_kl_loss = 0.0
epoch_discriminator_loss_real = 0.0
epoch_discriminator_loss_fake = 0.0
epoch_discriminator_loss = 0.0
epoch_generator_loss = 0.0

reporter = rp.Reporter('./logs/vae_gan', './logs/snapshots')

flip = False
data, _ = next(iter(data_loader_mnist))
print(dt.test_discriminator(discriminator, data.view(batch_size, -1)))
for epoch in range(num_epochs):
    if epoch % skip_iteration == 0:
        flip = not flip

    print(f"Epoch [{epoch}/{num_epochs}]")
    if(epoch > 0):
        reporter.add_scalar('MSE Loss', epoch_mse_loss, epoch)
        reporter.add_scalar('KL Loss', epoch_kl_loss, epoch)
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
        ###Train discriminator with real data###
        real_image = data.view(batch_size, -1)

        discriminator.zero_grad()
        #Expect real image to be classified as 1
        real_image_label = torch.ones(batch_size, 1).to(device)
        output_real = discriminator(real_image)

        real_data_error = nn.L1Loss()(output_real, real_image_label)
        real_data_error.backward(retain_graph=True)

        ###Train discriminator with fake data###
        fake_image = autoencoder(real_image)
        #Expect fake image to be classified as 0
        fake_image_label = torch.zeros(batch_size, 1).to(device)
        output_fake = discriminator(fake_image)

        fake_data_error = nn.L1Loss()(output_fake, fake_image_label)
        fake_data_error.backward(retain_graph=True)

        discriminator_error = real_data_error + fake_data_error

        epoch_discriminator_loss += discriminator_error
        epoch_discriminator_loss_real += real_data_error
        epoch_discriminator_loss_fake += fake_data_error

        if flip:
            discriminator_optimizer.step()

        # Train VAE on MSE and KL divergence
        autoencoder.zero_grad()
        x_hat = autoencoder(real_image)

        shaped_real_image = real_image.view(batch_size, 1, 28, 28)
        MSE = F.mse_loss(x_hat, shaped_real_image, reduction='mean')
        epoch_mse_loss += MSE
        epoch_kl_loss +=  autoencoder.encoder.kl
        vae_loss = MSE + (kl_annealer(epoch) * autoencoder.encoder.kl)
        vae_loss.backward(retain_graph=True)
        vae_optimizer.step()

        # Train VAE on discriminator output
        autoencoder.zero_grad()
        x_hat2 = autoencoder(real_image)
        discriminator_guess = discriminator(x_hat2)
        generator_error = nn.L1Loss()(discriminator_guess, real_image_label) * gan_annealer(epoch)
        epoch_generator_loss += generator_error
        generator_error.backward()

        if not flip:
            vae_optimizer.step()

    epoch_mse_loss /= num_mini_batches
    epoch_kl_loss /= num_mini_batches
    epoch_discriminator_loss /= num_mini_batches
    epoch_discriminator_loss_real /= num_mini_batches
    epoch_discriminator_loss_fake /= num_mini_batches
    epoch_generator_loss /= num_mini_batches

# viz.plot_latent_encoder(encoder, data_loader_mnist)
# viz.plot_generated(generator, z_dim, 20)
reporter.close()


# Set the autoencoder to evaluation mode
autoencoder.eval()

# Generate a batch of data
data, _ = next(iter(data_loader_mnist))
data = data.to(device)

# Reconstruct the data using the autoencoder
reconstructed_data = autoencoder(data)

# Convert the reconstructed data to numpy array
reconstructed_data = reconstructed_data.detach().cpu().numpy()

import matplotlib.pyplot as plt

# Create a figure to display the reconstructed samples
fig, axs = plt.subplots(len(data), 2, figsize=(8, 2*len(data)))

for i in range(len(data)):
    original_sample = data[i].detach().cpu().numpy()
    reconstructed_sample = reconstructed_data[i]

    # Plot the original sample
    axs[i, 0].imshow(original_sample.reshape(28, 28), cmap='gray')
    axs[i, 0].set_title("Original Sample")

    # Plot the reconstructed sample
    axs[i, 1].imshow(reconstructed_sample.reshape(28, 28), cmap='gray')
    axs[i, 1].set_title("Reconstructed Sample")

    # Remove the axis labels
    axs[i, 0].axis('off')
    axs[i, 1].axis('off')

# Adjust the spacing between subplots
plt.tight_layout()

# Save the figure
plt.savefig('./image.png')

# Show the figure
plt.show()

