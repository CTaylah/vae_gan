from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

device = 'cpu'

def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.savefig('latent.png')

def plot_latent_encoder(encoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.savefig('latent.png')
    plt.clf()


def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])
    plt.savefig('reconstructed.png')

def plot_reconstructed_decoder(decoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])
    plt.savefig('reconstructed.png')


def plot_generated(decoder, latent_dim, num_images):
    z = torch.randn(num_images, latent_dim).to(device)
    generated_images = decoder(z)

    generated_images = generated_images.cpu().detach().numpy()

    fig, axes = plt.subplots(1, num_images, figsize=(20, 2))
    for ax, img in zip(axes, generated_images):
        ax.imshow(img.reshape(28, 28), cmap='gray')
        ax.axis('off')

    plt.imshow(img, cmap='gray')
    plt.savefig('generated.png')

def save_tensor_as_ppm(tensor, filename, scale_factor=5):
    # Convert the tensor to a PIL image
    transform = transforms.ToPILImage()
    tensor = tensor.reshape(28, 28)
    image = transform(tensor)

    # Scale up the image
    width, height = image.size
    new_width = 28 * scale_factor
    new_height = 28 * scale_factor
    scaled_image = image.resize((new_width, new_height))

    # Save the scaled image as a ppm file
    scaled_image.save(filename, 'PPM')
