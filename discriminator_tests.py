import torch

# Outputs what percentage of the time the discriminator correctly identifies the real image
# The discriminator is a PyTorch neural network
# Reconstructions is a PyTorch tensor of images
def test_discriminator(discriminator, images, real: bool=True):
    # Initialize the count of correct predictions
    output = discriminator(images)
    real_image_label = torch.ones(images.shape[0], 1)
    fake_image_label = torch.zeros(images.shape[0], 1)
    if real:
        return torch.sum(output > 0.5).item() / images.shape[0]
    