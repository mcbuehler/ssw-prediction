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
    """A class that trains an autoencoder on our original data. It can scale the
    data per feature in [-1, 1] or not scale them at all. The autoencoder is a
    (denoising) two layer one. There are separate methods for training and
    producing the encondings when you have already trained a model.
    Attributes
    ----------
        feature_count: int
            The number of features we have
    """

    feature_count = 3

    def __init__(self, batch_size, cuda, scale, flatten):
        """The constructor of the class. Also sets the random seed and the least
        available GPU if this is available.

        Parameters
        ----------
            batch_size: int
                The batch size
            cuda: bool
                A boolean to run on CPU or GPU when the latter is available
            scale: bool
                Choose if you are going to scale your features or not (the
                scaling is done per feature)
            flatten: bool
                A flag to know if you are getting flatten data or a tensor

        """
        self.batch_size = batch_size
        self.cuda = cuda
        self.scale = scale
        self.flatten = flatten

        if self.cuda:
            set_gpu()

        SetSeed().set_seed()

    def _scale_per_variable(self, data, scalers_exist, scalers=None,
                            scale_range=(-1, 1)):
        """A function that either fits a scaler per variable and then transforms
        the data per variable as well during the training or just tranforms the
        data per variable in case we want to produce the embeddings.

        Parameters
        ----------
            data: numpy array
                The original data with shape [num_data, feature_count *
                dimensionality]
            scalers_exit: bool
                A flag as to now if the scalers exist or they have to be fitted
            scalers: list or None
                A python list that will have the fitted scalers or nothing in
                case the scalers are produced from scratch
            scale_range: tuple
                The range the scaling will be done in.

        Returns
        -------
            scalers: list
                A list of scalers that have either been fitted or they are the
                same as the ones passed as a parameter
            data: numpy array
                The scaled data with shape [num_data, feature_count *
                dimensionality]
        """
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
        """A function that preprocesses the data either in the training or in
        the produce_encodings phase. It flattens the data from shape [num_data,
        feature_count, dimensionality] to [num_data, feature_count *
        dimensionality] and then scales them. Finally it returns a DataLoader
        class for a model to be fitted.
        Parameters
        ----------
            temp_data: numpy array
                The original data with dimensions [num_data, num_features,
                dimensionality]
            mode: string
                The mode that the preprocessing is going to be run ('train' or
                'eval')
            scalers: list or None
                A python list that will have the fitted scalers or nothing in
                case the scalers are produced from scratch

        Returns
        -------
            scalers: list
                A list of scalers that have either been fitted or they are the
                same as the ones passed as a parameter
            dataloader: DataLoader class
                A dataloader that is ready to be fed into a NN
        """
        if self.flatten:
            _, self.feature_count, self.dimensionality = \
                temp_data.shape
            self.input_dim = self.feature_count * self.dimensionality

            # bring data from format (N, FC, D) to (N, FC * D)
            data = [np.hstack(
                (temp_data[i, j-2], temp_data[i, j-1], temp_data[i, j]))
                for i in range(len(temp_data))
                for j in range(len(temp_data[i]))
                if j % self.feature_count == self.feature_count - 1
                ]
            data = np.array(data)
        else:
            _, self.input_dim = temp_data.shape
            self.dimensionality = self.input_dim
            data = np.copy(temp_data)

        # decide if you are going to scale or not
        if self.scale:
            if scalers is None:
                scalers, data = self._scale_per_variable(data=data,
                                                         scalers_exist=False)
            else:
                _, data = self._scale_per_variable(data=data,
                                                   scalers_exist=True,
                                                   scalers=scalers)
        else:
            scalers = []

        # return the correct dataloader based on your mode
        if mode == 'train':
            dataloader = DataLoader(data, batch_size=self.batch_size)
        else:
            dataloader = DataLoader(data, batch_size=temp_data.shape[0])

        return scalers, dataloader

    def load_data(self, definition, path, cutoff_point, prediction_start_day,
                  prediction_interval, features_interval):
        """This function uses the FixedWindowPredictionSet class in order to
        return the data.

        Parameters
        ----------
            definition: string
                the definition that you will get the labels for
            path: string
                the path of the data (real or simulated)
            cutoff_point: int
                the maximum cutoff_point where you will look your time series
            prediction_start_day: int
                the day you are going to start your predictions for
            prediction_interval: int
                the interval you are going to make predictions for
            features_interval: int
                the number of days in the past that you will look the time
                series before the cutoff_point
        Returns
        -------
            data: numpy array
                A numpy array of shape [num_data, num_features, dimensionality]
                that contains the data
        """

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
        """Trains the two layer autoencoder. It sets all the necessary
        parameters for initializing the NN and then it trains it and returns
        the trained one.
        Parameters
        ----------
            num_epochs: int
                The number of epochs the network is going to be trained for
            optimizer: torch.optim function
                The optimizer that is going to be used for updating the NN
            learning_rate: float
                The learning rate of the optimizer
            denoising: bool
                Flag to decide if you are going to train the denoising version
                of the autoencoder
        Returns
        """
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

        trained_encondings = []
        for data in dataloader:
            data = data.detach()
            encondings, output = model(data.float())
            if self.cuda:
                temp_encondings = encondings.detach().cpu().numpy()
            else:
                temp_encondings = encondings.detach().numpy()
            for enconding in temp_encondings:
                trained_encondings.append(enconding)

        return model, np.array(trained_encondings)

    def produce_encodings(self, model, data, scalers):
        """Produces the encondings by getting an already trained model, the
        scalers used for the preprocessing of the training data (if they exist)
        and the validation/test data in their original format.

        Parameters
        ----------
            model: torch.nn.Module class
                A trained two-layers autoencoder
            data: numpy array
                the validation/test data with dimensions [num_data,
                num_features, dimensionality]
            scalers: list
                A list of scalers that have either been fitted (or maybe empty)
        Returns
        -------
            encondings: numpy array
                The encondings with size [num_data, enconding_dimension]
        """
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
                               args.scale, flatten=True)
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
    model, trained_encondings = test.train(args.epochs, optimizer, 0.01,
                                           args.denoising, dataloader)
    encondings = test.produce_encodings(model, data[2000:], scalers)
