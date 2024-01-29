import torch; torch.manual_seed(9)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision

from torch.utils.tensorboard import SummaryWriter


input_size = 784
device = 'cpu'

class Encoder(nn.Module):
    def __init__(self, sequential):
        super().__init__()
        self.layers = sequential
        self.mu_layer = nn.Linear(sequential[-2].in_features, sequential[-2].out_features)
        self.log_variance_layer = nn.Linear(sequential[-2].in_features, sequential[-2].out_features)
        self.layers.pop(len(self.layers)-1)
        self.layers.pop(len(self.layers)-1)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

        #Fishy
        self.mu = None
        self.log_variance = None
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.layers.forward(x)
        self.mu = self.mu_layer(x)
        self.log_variance = self.log_variance_layer(x)
        #Reparametrization trick
        eps = self.N.sample(self.mu.shape).to(device)
        variance = torch.exp(self.log_variance)
        z = self.mu + (variance * eps) 
        self.kl = torch.mean((variance**2 + self.mu**2 - self.log_variance - 1/2))
        return z
    
    def draw_sample(self, n):
        z = torch.randn(n, self.mu_layer.out_features).to(device)
        return self.decode(z)


class Decoder(nn.Module):
    def __init__(self, sequential):
        super().__init__()
        self.layers = sequential
    
    def forward(self, z):
        z = self.layers.forward(z)
        return z.reshape(-1, 1, 28, 28)


class VAE(nn.Module):
    def __init__(self, encoder_list, decoder_list):
        super(VAE, self).__init__()

        encoder_sequential = nn.Sequential()
        for i in range(len(encoder_list)-1):
            encoder_sequential.add_module(f'layer{i}', nn.Linear(encoder_list[i], encoder_list[i+1]))
            encoder_sequential.add_module(f'relu{i}', nn.ReLU())

        decoder_sequential = nn.Sequential()
        for i in range(len(decoder_list)-1):
            decoder_sequential.add_module(f'layer{i}', nn.Linear(decoder_list[i], decoder_list[i+1]))
            decoder_sequential.add_module(f'relu{i}', nn.ReLU())
        


        self.encoder = Encoder(encoder_sequential)
        self.decoder = Decoder(decoder_sequential)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, n):
        z = torch.randn(n, self.encoder.mu_layer.out_features).to(device)
        return self.decoder(z)

def vae_train(model, optimizer, data, epochs, verbose=False, writer:SummaryWriter=None):
    kl_epoch = epochs//2
    for  epoch in range(epochs):
        for x, y in data:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            mse = F.mse_loss(x_hat, x, reduction='sum')
            loss = mse
            if epoch > kl_epoch:
                loss = mse + model.encoder.kl
            loss.backward()
            optimizer.step()
        if verbose:
            writer.add_scalar('MSE/train', mse.item()/len(data.dataset), epoch)
            writer.add_scalar('KL/train', model.encoder.kl, epoch)
            writer.add_scalar('Loss/train', loss.item()/len(data.dataset), epoch)
            print(f'Epoch: {epoch}, Loss: {loss.item()/len(data.dataset)}')
    return model

