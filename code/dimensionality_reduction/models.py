from torch import nn
from torch.autograd import Variable
# base1: https://github.com/ShayanPersonal/stacked-autoencoder-pytorch/
# base2: https://github.com/L1aoXingyu/pytorch-beginner/


class TwoLayerAutoencoder(nn.Module):
    """A class to create a 4 layer autoencoder with one encoder and one decoder
    hidden layer. It initializes the structure of the NN using Pytorch
    functions and also implements the forward method for the forward pass of
    the network."""
    def __init__(self, input_dim, first_layer_dim, second_layer_dim,
                 activation, denoising):
        """The constructor of the class. It gets all the necessary parameters in
        order to construct a 2-layer autoencoder.

        Parameters
        ----------
            input_dim: int
                The input layer dimension
            first_layer_dim: int
                The first hidden layer dimension
            second_layer_dim: int
                The second hidden layer dimension
            activation: torch.nn instance
                The activation function used in the network
            denoising: bool
                Decide if the autoencoder is going to be noisy or not
        """
        super().__init__()
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
        """The function that implements the forward pass of the network.
        Parameters
        ----------
            x: torch.Variable
                The input data with dimensions [batch_size x dimensionality]

        Returns
        -------
            x_encoded: torch.Variable
                The encoded data with dimensions
                [batch_size x second_layer_dim]
            x_decoded: torch.Variable
                The reconstructed data with dimensions
                [batch_size x input_dim]
        """
        if self.denoising:
            x_noisy = x * (Variable(
                x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
            x_encoded = self.encoder(x_noisy)
        else:
            x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_encoded, x_decoded
