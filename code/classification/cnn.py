import os

import cnn_model
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from data_manager import DataManager
from dataset import DatapointKey as DPK
from sklearn.model_selection import train_test_split
import setGPU


class SSWDataset(data.Dataset):
    """
    Extends PyTorch's data.Dataset class to provide a dataset for PyTorch
    models. It creates a training and test dataset. It packs "wind_65",
    "temp_60_90" and "wind_60" as features and SSW_definition as yearly
    labels as prompted by the user
    """

    # A dictionary which indicates which time series is going
    # to be used in the classifier
    data_to_use = {
        DPK.CP07: [DPK.WIND_65, DPK.TEMP_60_90, DPK.WIND_60],
        DPK.UT: [DPK.WIND_65, DPK.TEMP_60_90, DPK.WIND_60],
        DPK.U65: [DPK.WIND_65, DPK.TEMP_60_90, DPK.WIND_60],
        DPK.ZPOL: [DPK.WIND_65, DPK.TEMP_60_90, DPK.WIND_60],
    }

    def __init__(self, file_path, label_type, train=True):
        """
        Initializer of SSWDataset
        :param file_path: Path to h5 file which contains labeled dataset
        :param label_type: Definition to be used for labeling (i.e. "CP07")
        :param train: Returns test data if False, returns training data if
        True
        """
        super(SSWDataset, self).__init__()

        data_type = SSWDataset.data_to_use[label_type]

        # Get data from data manager
        data_manager = DataManager(file_path)

        features = data_manager.get_data_for_variables(data_type)
        labels = np.any(data_manager.get_data_for_variable(label_type),
                        axis=1)

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2,
            stratify=labels, random_state=42)

        if train:
            self.data = X_train
            self.labels = y_train
        else:
            self.data = X_test
            self.labels = y_test

    def __getitem__(self, index):
        return (torch.from_numpy(self.data[index]).float(),
                self.labels[index].astype(int))

    def __len__(self):
        return self.data.shape[0]


class ConvNetClassifier():

    def __init__(self, file_path, label_type):
        """
        Initializer for ConvNetClassifier
        :param file_path: Path to h5 file which contains labeled dataset
        :param label_type: Definition to be used for labeling (i.e. "CP07")
        """
        # Device configuration
        device = torch.device('cuda:' + os.getenv("CUDA_VISIBLE_DEVICES")
                              if torch.cuda.is_available() else 'cpu')

        self.label_type = label_type
        self.train_dataset = SSWDataset(file_path, self.label_type)
        self.test_dataset = SSWDataset(file_path, self.label_type, train=False)

        # Number of channels in the CNN - number of features to use
        num_ts = len(SSWDataset.data_to_use[self.label_type])
        self.model = cnn_model.ConvNet(num_ts).to(device)

    def train(self, num_epochs=100, batch_size=16, learning_rate=0.0004):
        """
        Function to train the CNN.
        :param num_epochs: Number of Epochs to train the network
        :param batch_size: Batch size for training
        :param learning_rate: Learning rate for Adam optimizer
        """
        # Device configuration
        device = torch.device('cuda:' + os.getenv("CUDA_VISIBLE_DEVICES")
                              if torch.cuda.is_available() else 'cpu')

        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        self.model.train()

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=learning_rate)

        for epoch in range(num_epochs):
            for i, (winters, labels) in enumerate(train_loader):
                winters = winters.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self.model(winters)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def test(self):
        """
        Function to test the CNN.
        """
        # Device configuration
        device = torch.device('cuda:' + os.getenv("CUDA_VISIBLE_DEVICES")
                              if torch.cuda.is_available() else 'cpu')

        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  shuffle=False)

        self.model.eval()

        with torch.no_grad():
            correct = 0
            total = 0

            for winters, labels in test_loader:
                winters = winters.to(device)
                labels = labels.to(device)

                outputs = self.model(winters)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the {} test winters: {} %'.
                  format(len(test_loader.dataset), 100 * correct / total))


if __name__ == '__main__':
    path_preprocessed = os.getenv("DSLAB_CLIMATE_BASE_OUTPUT")
    filename = os.path.join(path_preprocessed, "data_labeled.h5")
    definitions = [DPK.CP07, DPK.UT, DPK.U65]

    for definition in definitions:
        classifier = ConvNetClassifier(filename, definition)
        classifier.train(num_epochs=100, batch_size=8, learning_rate=0.0004)
        classifier.test()
