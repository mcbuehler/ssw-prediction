import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 100
batch_size = 16
learning_rate = 0.001


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
        return (torch.from_numpy(self.data[index]).float(),
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


if __name__ == '__main__':
    filename = '../data/data_preprocessed_labeled.h5'
    definition = "CP07"

    train_dataset = SSWDataset(filename, definition)
    test_dataset = SSWDataset(filename, definition, train=False)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    model = ConvNet(len(SSWDataset.data_to_use[definition])).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the {} test winters: {} %'.format(len(test_loader), 100 * correct / total))
