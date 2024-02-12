

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
torch.manual_seed(42)

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
            nn.Linear(in_features, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.disc(x)



# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
vae_lr = 5e-5
gen_lr = 5e-5
dis_lr = 5e-5
z_dim = 266
image_dim = 28 * 28 * 1  # 784
batch_size = 128
num_epochs = 50
negative_slope = 0.01
# How many epochs to train the discriminator before changing the generator
generator_iterations = 1
# Vice versa
discriminator_iterations = 1


# autoencoder = VAE(z_dim, 512).to(device)
autoencoder = vae.VAE([image_dim, 512, 368, z_dim], [z_dim, 368, 512, image_dim])

discriminator = Discriminator(image_dim)


dataset_mnist = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
data_loader_mnist = torch.utils.data.DataLoader(dataset_mnist, batch_size=batch_size, shuffle=True)


dataset_mnist_test = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
data_loader_mnist_test = torch.utils.data.DataLoader(dataset_mnist_test, batch_size=batch_size, shuffle=True)

discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=dis_lr)
vae_optimizer = optim.Adam(autoencoder.parameters(), lr=gen_lr)

kl_annealer = annealer.Annealer(0.0, 0.1, 0.01, start_epoch=0)
gan_annealer = annealer.Annealer(0.0, 0.1, 0.01, start_epoch=0)

num_mini_batches = len(data_loader_mnist)

reporter = rp.Reporter('./logs/vae_gan', './logs/snapshots')
reporter.add_loss_labels(['MSE Loss', 'KL Loss', 'Discriminator Loss', 'Discriminator Loss Real', 'Discriminator Loss Fake', 'Generator Loss'])

flip = True
discriminator_counter = 0
generator_counter = 0
for epoch in range(num_epochs):
    if flip and discriminator_counter >= discriminator_iterations:
        flip = not flip
        discriminator_counter = 0
    elif not flip and generator_counter >= generator_iterations:
        flip = not flip
        generator_counter = 0



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
        # real_data_error.backward(retain_graph=True)

        ###Train discriminator with fake data###
        fake_image = autoencoder(real_image) + (0.1 * torch.randn(batch_size, image_dim).view(batch_size, 1, 28, 28).to(device))
        #Expect fake image to be classified as 0
        fake_image_label = torch.zeros(batch_size, 1).to(device)
        output_fake = discriminator(fake_image)

        fake_data_error = nn.L1Loss()(output_fake, fake_image_label)
        # fake_data_error.backward(retain_graph=True)

        if flip:
            discriminator_error = real_data_error + fake_data_error
            discriminator_error.backward(retain_graph=True)
            discriminator_optimizer.step()

        reporter.accumulate_loss('Discriminator Loss', discriminator_error)
        reporter.accumulate_loss('Discriminator Loss Real', real_data_error)
        reporter.accumulate_loss('Discriminator Loss Fake', fake_data_error)

            
        # Train VAE on MSE and KL divergence
        autoencoder.zero_grad()
        x_hat = autoencoder(real_image)

        shaped_real_image = real_image.view(batch_size, 1, 28, 28)
        MSE = F.mse_loss(x_hat, shaped_real_image, reduction='mean')

        reporter.accumulate_loss('MSE Loss', MSE)
        reporter.accumulate_loss('KL Loss', autoencoder.encoder.kl)

        # Train VAE on discriminator output
        discriminator_guess = discriminator(x_hat)
        generator_error = nn.L1Loss()(discriminator_guess, real_image_label)

        reporter.accumulate_loss('Generator Loss', generator_error)

        vae_loss = (MSE + (kl_annealer(epoch) * autoencoder.encoder.kl)) + (gan_annealer(epoch) * generator_error)
        if not flip:
            vae_loss.backward()
            vae_optimizer.step()

    print(f"Epoch [{epoch}/{num_epochs}]")
    # if(epoch > 0):
    reporter.write_losses(epoch, len(data_loader_mnist))
    reporter.zero_losses()


    if flip:
        discriminator_counter += 1
    else:
        generator_counter += 1

reporter.close()


# Set the autoencoder to evaluation mode
autoencoder.eval()

# Generate a batch of data
data, _ = next(iter(data_loader_mnist_test))
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
plt.savefig(reporter.file_folder + '/reconstructed_samples.png')
plt.savefig('./reconstructed_samples.png')

reporter.add_image('Reconstructed Samples', reporter.file_folder + '/reconstructed_samples.png')
# Show the figure
