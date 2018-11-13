# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py

torch.manual_seed(1)

######################################################################

BATCH_SIZE = 1
SEQ_LEN = 210
INPUT_DIM = 1
HIDDEN_DIM = 1  # what is this exactly?


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

        for day in sequence_data:
            # Step through the sequence one element at a time.
            # after each step, hidden contains the hidden state.
            out, self.hidden = self.lstm(day.view(1, 1, -1), self.hidden)

        ssw_space = self.hidden2ssw(out.view(len(sequence_data), -1))
        ssw_score = F.log_softmax(ssw_space, dim=1)
        return ssw_score


def load_input_data():
    file = h5py.File("data/labeled_output/data_preprocessed_labeled.h5", "r")
    # TODO: implement
    # Output should be iterable with features, label
    return list(), list()

data_train, data_test = load_input_data()
exit()

model = SSWPredictorLSTM(INPUT_DIM, HIDDEN_DIM)
# negative log likelihood loss
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for features, label in data_train:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 3. Run our forward pass.
        tag_scores = model(features)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, label)
        loss.backward()
        optimizer.step()
