import torch
import numpy as np
import argparse
from torch import nn
from torch.utils.data import DataLoader
from utils.set_seed import SetSeed
from prediction.prediction_set import FixedWindowPredictionSet
from utils.set_gpu import set_gpu
from sklearn.preprocessing import MinMaxScaler
from torch.optim import SGD
from dimensionality_reduction.models import TwoLayerAutoencoder


class AutoEncoderTraining:
    def __init__(self, batch_size, cuda, scale):
        self.batch_size = batch_size
        self.cuda = cuda
        self.scale = scale

        if self.cuda:
            set_gpu()

        SetSeed().set_seed()

    def scale_per_variable(self, data, scalers_exist, scalers=None,
                           scale_range=(-1, 1)):
        # normalize data to [-1, 1]
        if scalers is None:
            scalers = []
        for i in range(data.shape[1]//self.dimensionality):
            if not scalers_exist:
                scaler = MinMaxScaler(feature_range=scale_range)
                scaler.fit(data[i * self.dimensionality:
                                i * self.dimensionality +
                                self.dimensionality])
                scalers.append(scaler)

            data[i * self.dimensionality: i * self.dimensionality +
                 self.dimensionality] = scalers[i].transform(
                            data[i * self.dimensionality:
                                 i * self.dimensionality +
                                 self.dimensionality])
        return scalers, data

    def preprocessing(self, temp_data, mode, scalers=None):
        _, self.feature_count, self.dimensionality = \
                temp_data.shape
        self.input_dim = self.feature_count * self.dimensionality

        data = [np.hstack(
            (temp_data[i, j-2], temp_data[i, j-1], temp_data[i, j]))
            for i in range(len(temp_data))
            for j in range(len(temp_data[i]))
            if j % self.feature_count == self.feature_count - 1
            ]
        data = np.array(data)

        if self.scale:
            if scalers is None:
                scalers, data = self.scale_per_variable(data=data,
                                                        scalers_exist=False)
            else:
                _, data = self.scale_per_variable(data=data,
                                                  scalers_exist=True,
                                                  scalers=scalers)
        else:
            scalers = []

        if mode == 'train':
            dataloader = DataLoader(data, batch_size=self.batch_size)
        else:
            dataloader = DataLoader(data, batch_size=temp_data.shape[0])

        return scalers, dataloader

    def load_data(self, definition, path, cutoff_point, prediction_start_day,
                  prediction_interval, features_interval):
        self.prediction_set = FixedWindowPredictionSet(
                definition,
                path,
                cutoff_point,
                prediction_start_day,
                prediction_interval,
                features_interval
                )

        # returns the data in format (N, FC, D)
        data = self.prediction_set.get_features()
        return data

    def train(self, num_epochs, optimizer, learning_rate,
              denoising, dataloader):
        first_layer_dim = self.input_dim // 2
        second_layer_dim = first_layer_dim // 2

        activation = nn.Sigmoid()

        model = TwoLayerAutoencoder(self.input_dim, first_layer_dim,
                                    second_layer_dim, activation, denoising)
        if self.cuda:
            model.cuda()

        optimizer = optimizer(model.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()
        for epoch in range(num_epochs):
            for data in dataloader:
                data = data.float()
                if self.cuda:
                    data.cuda()
                encoding, output = model(data)
                loss = loss_function(output, data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, num_epochs, loss.item()))

        return model

    def produce_encodings(self, model, data, scalers):
        _, dataloader = self.preprocessing(temp_data=data, mode='eval',
                                           scalers=scalers)
        for data in dataloader:
            encondings, output = model(data.float())

        if self.cuda:
            encondings = encondings.detach().cpu().numpy()
        else:
            encondings = encondings.detach().numpy()
        return encondings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature extraction using \
            autoencoders')
    parser.add_argument(
            "-d",
            "--definition",
            choices=('CP07', 'U65', 'U&T'),
            help="Choose the definition that you want to run classification",
            action="store",
            default="CP07"
           )
    parser.add_argument(
            "-sp",
            "--path",
            help="Choose the input relative path where the simulated data are",
            action="store",
            default="data/simulated_data_labeled.h5"
            )
    parser.add_argument(
            "-bs",
            "--batch_size",
            help="Choose the batch size",
            type=int,
            action="store",
            default=16
            )
    parser.add_argument(
            "-cp",
            "--cutoff_point",
            help="Choose the cutoff point of the time series",
            type=int,
            action="store",
            default=90
            )
    parser.add_argument(
            "-fi",
            "--features_interval",
            help="Choose the interval where you will calculate features",
            type=int,
            action="store",
            default=30
            )
    parser.add_argument(
            "-sd",
            "--prediction_start_day",
            help="Choose the day you will start making predictions for",
            type=int,
            action="store",
            default=7
            )
    parser.add_argument(
            "-pi",
            "--prediction_interval",
            help="Choose the interval you are going to make predictions for",
            type=int,
            action="store",
            default=7
            )
    parser.add_argument(
            "-e",
            "--epochs",
            help="Choose the number of epochs your will train",
            type=int,
            action="store",
            default=300
            )
    parser.add_argument(
            "-n",
            "--denoising",
            help="Choose if you are going to train the denoising version",
            action="store_true",
            default=True
            )
    parser.add_argument(
            "-s",
            "--scale",
            help="Choose if you are going to scale the features",
            action="store_true",
            default=False
            )
    args = parser.parse_args()
    test = AutoEncoderTraining(args.batch_size, torch.cuda.is_available(),
                               args.scale)
    data = test.load_data(
            definition=args.definition,
            path=args.path,
            cutoff_point=args.cutoff_point,
            prediction_start_day=args.prediction_start_day,
            prediction_interval=args.prediction_interval,
            features_interval=args.features_interval,
            )
    optimizer = SGD
    scalers, dataloader = test.preprocessing(data[:2000], mode='train')
    model = test.train(args.epochs, optimizer, 0.01,
                       args.denoising, dataloader)
    encondings = test.produce_encodings(model, data[2000:], scalers)
    print(encondings)
