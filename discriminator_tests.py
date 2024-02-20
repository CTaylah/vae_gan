import torch
import torch.nn as nn
import torch.optim as optim


from reporter import Reporter
from discriminator import Discriminator
from roc_plotting import roc_plot

image_dim = 784
device = 'cpu'

def test_discriminator(generator: nn.Module, data_loader,reporter: Reporter, training_epochs=2, lr=5e-5):
    discriminator = Discriminator(784)
    optimizer = discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(training_epochs):
        print(f"Meta Discriminator epoch: {epoch}")
        for i, (data, label) in enumerate (data_loader):
 
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
            fake_image = generator(real_image) 
            #Expect fake image to be classified as 0
            fake_image_label = torch.zeros(batch_size, 1).to(device)
            output_fake = discriminator(fake_image)

            fake_data_error = nn.L1Loss()(output_fake, fake_image_label)
            # fake_data_error.backward(retain_graph=True)

            discriminator_error = real_data_error + fake_data_error
            discriminator_error.backward()
            discriminator_optimizer.step()

    roc_plot(discriminator, generator, data_loader, device, reporter)
    

    
    

