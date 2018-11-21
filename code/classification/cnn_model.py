import torch.nn as nn


class ConvNet(nn.Module):
    """
    Pytorch CNN model for yearly SSW classification.
    """

    def __init__(self, input_channels, k1=15, k2=10, filt1=16, filt2=32):
        """
        Initializer for the CNN model.

        :param input_channels: Number of input channels, number of yearly
        time_series as features example: "wind_65" "temp_60_90" and "wind_60"
        are the features, 3 should be inputted
        :param k1: kernel size for first convolutional layer
        :param k2: kernel size for second convolutional layer
        :param filt1: number of filters for first convolutional layer
        :param filt2: number of filters for second convolutional layer
        """
        super(ConvNet, self).__init__()

        # Number of days in winter
        NUM_DAYS = 210

        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, filt1, kernel_size=k1),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv1d(filt1, filt2, kernel_size=k2),
            nn.ReLU())

        self.fc = nn.Linear((NUM_DAYS - k1 - k2 + 2) * filt2, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
