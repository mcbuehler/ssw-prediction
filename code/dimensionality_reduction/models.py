# import torch
from torch import nn
from torch.autograd import Variable

# base1: https://github.com/ShayanPersonal/stacked-autoencoder-pytorch/
# base2: https://github.com/L1aoXingyu/pytorch-beginner/


class TwoLayerAutoencoder(nn.Module):
    def __init__(self, input_dim, first_layer_dim, second_layer_dim,
                 activation, denoising=False):
        super(TwoLayerAutoencoder, self).__init__()
        self.denoising = denoising
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, first_layer_dim),
            activation,
            nn.Linear(first_layer_dim, second_layer_dim),
            activation)
        self.decoder = nn.Sequential(
            nn.Linear(second_layer_dim, first_layer_dim),
            activation,
            nn.Linear(first_layer_dim, input_dim)
            )

    def forward(self, x):
        if self.denoising:
            x_noisy = x * (Variable(
                x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
            x_encoded = self.encoder(x_noisy)
        else:
            x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_encoded, x_decoded
