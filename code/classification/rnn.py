# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from sklearn.model_selection import train_test_split

from data_manager import DataManager

torch.manual_seed(1)

######################################################################
# Code based on
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

LEARNING_RATE = 0.1
N_EPOCHS = 300
BATCH_SIZE = 128
SEQ_LEN = 210
INPUT_DIM = 1
HIDDEN_DIM = 1  # what is this exactly?
INPUT_FILE = os.getenv(
    "DSLAB_CLIMATE_LABELED_DATA")
# local: "../data/labeled_output/data_preprocessed_labeled.h5"


class SSWPredictorLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SSWPredictorLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2ssw = nn.Linear(hidden_dim, 1)  # 1 output dimension
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, BATCH_SIZE, self.hidden_dim),
                torch.zeros(1, BATCH_SIZE, self.hidden_dim))

    def forward(self, sequence_data):
        assert len(sequence_data) > 0
        sequence_data = np.transpose(sequence_data, (1, 0,))
        sequence_data = np.reshape(sequence_data, (210, -1, 1))
        sequence_data = torch.FloatTensor(sequence_data)

        out, self.hidden = self.lstm(sequence_data, self.hidden)

        ssw_space = self.hidden2ssw(sequence_data)
        ssw_score = F.log_softmax(ssw_space, dim=0)
        return ssw_score


def load_input_data():
    dm = DataManager(INPUT_FILE)
    train_feature = dm.get_data_for_variable("wind_60")
    train_label = dm.get_data_for_variable("CP07")
    # Output should be iterable with features, label for train and test
    # shape (1372, 2, 210)
    return train_feature, train_label


all_features, all_labels = load_input_data()

train_features, test_features, train_labels, test_labels = train_test_split(
    all_features, all_labels, test_size=0.2, random_state=42)
N = len(all_features)

model = SSWPredictorLSTM(INPUT_DIM, HIDDEN_DIM)
# negative log likelihood loss
criterion = nn.NLLLoss()
# loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)


def sample_indices(n_sampled, n):
    return np.random.random_integers(0, n - 1, n_sampled)


for epoch in range(
        N_EPOCHS):
    # again, normally you would NOT do 300 epochs, it is toy data
    print(epoch)
    loss = 0

    indices = sample_indices(BATCH_SIZE, len(train_features))

    features = torch.FloatTensor(train_features[indices, :])
    labels = train_labels[indices, :].astype(np.int)
    labels = torch.LongTensor(np.transpose(labels, (1, 0)))

    # Step 1. Remember that Pytorch accumulates gradients.
    # We need to clear them out before each instance
    model.zero_grad()

    # Also, we need to clear out the hidden state of the LSTM,
    # detaching it from its history on the last instance.
    model.hidden = model.init_hidden()

    # Step 3. Run our forward pass.
    tag_scores = model(features)
    for i in range(BATCH_SIZE):
        predictions_i = tag_scores[:, i, :]
        labels_i = labels[:, i]
        loss += criterion(predictions_i, labels_i)

    print(loss)
    loss.backward()

    optimizer.step()
