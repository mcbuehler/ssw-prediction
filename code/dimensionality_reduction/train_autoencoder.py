import sys
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.set_seed import SetSeed
from prediction.prediction_set import FixedWindowPredictionSet
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from dimensionality_reduction.models import TwoLayerAutoencoder


class AutoEncoderTraining:
    def __init__(self, batch_size, cuda):
        self.batch_size = batch_size
        self.cuda = cuda

        SetSeed().set_seed()

    def _prepare_data(self, temp_data, scale_range=(-1, 1)):
        self.feature_count = temp_data.shape[1]
        print(temp_data.shape)
        data = [np.hstack(
            (temp_data[i , j], temp_data[i, j-1], temp_data[i, j-2]))
            for i in range(len(temp_data)) 
            for j in range(len(temp_data[i]))
            if j % self.feature_count == self.feature_count - 1
            ]
        data = np.array(data)
        self.input_dim = data.shape[1]

        # normalize data to [-1, 1]
        scaler = MinMaxScaler(feature_range=scale_range)
        scaler.fit(data)
        data = scaler.transform(data)
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def load_data(self, definition, path, cutoff_point, week_interval,
                  features_interval):
        self.prediction_set = FixedWindowPredictionSet(
                definition,
                path,
                cutoff_point,
                week_interval,
                features_interval
                )

        # returns the data in format (N, FC, D)
        data = self.prediction_set.get_features()
        return data
    
    def train(self, num_epochs, optimizer, learning_rate,
              first_layer_dim, second_layer_dim, denoising, dataloader):
        model = TwoLayerAutoencoder(self.input_dim, first_layer_dim,
                                    second_layer_dim, denoising)
        optimizer = optimizer(model.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()
        for epoch in range(num_epochs):
            for data in dataloader:
                data = data.float()
                encoding, output = model(data)
                loss = loss_function(output, data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, num_epochs, loss.data[0]))

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='A prediction scheme \
    #         using feature engineering and the XGBoostClassifier')
    # parser.add_argument(
    #         "-d",
    #         "--definition",
    #         choices=('CP07', 'U65', 'U&T'),
    #         help="Choose the definition that you want to run classification",
    #         action="store",
    #         default="CP07"
    #        )
    # parser.add_argument(
    #         "-sp",
    #         "--simulated_path",
    #         help="Choose the input relative path where the simulated data are",
    #         action="store",
    #         default="data/simulated_data_labeled.h5"
    #         )
    # parser.add_argument(
    #         "-rp",
    #         "--real_path",
    #         help="Choose the input relative path where the real data are",
    #         action="store",
    #         default="data/real_data_labeled.h5"
    #         )
    test = AutoEncoderTraining(16, torch.cuda.is_available)
    data = test.load_data(
            'CP07',
            'data/simulated_data_labeled.h5',
            90,
            1,
            7
            )
    dataloader = test._prepare_data(data)
    test.train(100, SGD, 0.01, 100, 50, False, dataloader)
