import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SSWDataset(data.Dataset):
    data_to_use = {
        'CP07': ['wind_60'],
        'U&T': ['temp_80_90', 'temp_60_70', 'wind_60'],
        'U65': ['wind_65'],
        'ZPOL_temp': ['temp_60_90']
    }

    def __init__(self, file_path, label_type, train=True):
        super(SSWDataset, self).__init__()

        f = h5py.File(file_path, 'r')
        keys = list(f.keys())
        data_type = SSWDataset.data_to_use[label_type]

        features = np.array(
            [[f[key][data_field] for data_field in data_type] for key in keys])
        labels = np.array([any(f[key][label_type]) for key in keys])

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
        return (torch.from_numpy(self.data[index] - np.mean(self.data[index])).float(),
                self.labels[index].astype(int))

    def __len__(self):
        return self.data.shape[0]


class ConvNet(nn.Module):
    def __init__(self, input_channels):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.fc = nn.Linear(52 * 32, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class ConvNetClassifier():

    def __init__(self, file_path, label_type, num_epochs=100,
                 batch_size=16, learning_rate=0.001):

        self.label_type = label_type
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        train_dataset = SSWDataset(file_path, self.label_type)
        test_dataset = SSWDataset(file_path, self.label_type, train=False)

        # Data loader
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False)

        num_ts = len(SSWDataset.data_to_use[self.label_type])
        self.model = ConvNet(num_ts).to(device)

    def train(self):

        self.model.train()

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)

        total_step = len(self.train_loader)

        for epoch in range(self.num_epochs):
            for i, (winters, labels) in enumerate(self.train_loader):
                winters = winters.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self.model(winters)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, self.num_epochs, i + 1,
                              total_step, loss.item()))

    def test(self):

        self.model.eval()

        with torch.no_grad():
            correct = 0
            total = 0

            for winters, labels in self.test_loader:
                winters = winters.to(device)
                labels = labels.to(device)

                outputs = self.model(winters)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the {} test winters: {} %'.
                  format(len(self.test_loader.dataset), 100 * correct / total))


if __name__ == '__main__':
    filename = '../data/data_preprocessed_labeled.h5'
    definition = "CP07"

    classifier = ConvNetClassifier(filename, definition)
    classifier.train()
    classifier.test()

    # CP07 %98.5 ---> 85 percent with centering
    # U65 %94.9
    # ZPOL_temp %53
    # with normalization 97.45
    #  ---> centering causes a bias, beware
    # U&T %93.45
