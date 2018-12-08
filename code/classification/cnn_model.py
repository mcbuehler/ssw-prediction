import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.enums import Classifier


class ConvNet(nn.Module):
    """
    Pytorch CNN model for yearly SSW classification.
    """

    def __init__(self, input_channels, k1=15, k2=20, filt1=32, filt2=64,
                 drop1=0.4, drop2=0.4, number_of_days=210):
        """
        Initializer for the CNN model.

        :param input_channels: Number of input channels, number of yearly
        time_series as features example: "wind_65" "temp_60_90" and "wind_60"
        are the features, 3 should be inputted
        :param k1: kernel size for first convolutional layer
        :param k2: kernel size for second convolutional layer
        :param filt1: number of filters for first convolutional layer
        :param filt2: number of filters for second convolutional layer
        :param drop1: dropout rate for first convolutional layer
        :param drop2: dropout rate for second convolutional layer
        """
        super(ConvNet, self).__init__()

        # Number of days in winter
        self.number_of_days = number_of_days

        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, filt1, kernel_size=k1),
            nn.ReLU(),
            nn.Dropout(p=drop1))

        self.layer2 = nn.Sequential(
            nn.Conv1d(filt1, filt2, kernel_size=k2),
            nn.ReLU(),
            nn.Dropout(p=drop2))

        self.fc = nn.Linear((number_of_days - k1 - k2 + 2) * filt2, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class ConvnetPoolingOverTime(nn.Module):

    def __init__(self, input_channels, Ks=(15, 10), filt=16, drop=0.4):
        """
        Initializer for ConvnetPoolingOverTime

        :param input_channels:Number of input channels, number of yearly
        time_series as features example: "wind_65" "temp_60_90" and "wind_60"
        are the features, 3 should be inputted
        :param Ks: list of integers for multiple kernel size
        :param filt: number of filters for the convolutional layer
        :param drop: dropout rate for the convolutional layer
        """
        super(ConvnetPoolingOverTime, self).__init__()

        self.convs1 = nn.ModuleList(
            [nn.Conv1d(input_channels, filt, kernel_size=K) for K in Ks])

        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(len(Ks) * filt, 2)

    def forward(self, x):
        out = [F.relu(conv(x)) for conv in self.convs1]
        out = [F.max_pool1d(i, i.size(2)) for i in out]
        out = torch.cat(out, 1).squeeze(2)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def get_cnn_classes():
    """
    Returns a dictionary of all CNN model classes included in this file
    :return: dictionary of classifier identifiers (utils.enums.Classifier)
    and CNN classes
    """
    return {
        Classifier.cnn: ConvNet,
        Classifier.cnn_max_pool: ConvnetPoolingOverTime
    }
