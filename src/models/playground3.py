import torch
import torch.nn as nn

from random import random


DROPOUT_RATE = 0.05
HIDDEN_SIZE = 128
NUM_LAYERS = 30
SKIP_RATE = 0.05


class Net(nn.Module):
    '''
    params:
                obs_size -- input size for input layer.
        hidden_size -- layer sizes.
        n_actions -- possible outputs (number of classes).
    '''
    def __init__(self, obs_size, hidden_size, n_actions,
            dropout_rate=DROPOUT_RATE, skip_rate=SKIP_RATE,
            num_layers=NUM_LAYERS):
        # XXX: included dropout as parameter option
        super(Net, self).__init__()
        self.sm = nn.Softmax(dim=1) # TODO: Verify not used; remove

        # Save parameters
        self.skip_rate = skip_rate

        # Standard compute unit.
        self.linear_relu_dropout_unit = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate))

        # Input layer
        self.input = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU())
        # Standard compute layer
        self.skipConnections = nn.ModuleList(
                [self.linear_relu_dropout_unit for _ in range(num_layers - 1)])
                # Subtract 1 layers for output layer.

        # Output layer
        self.out = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = self.input(x)
        for y in self.skipConnections:
           # Perform skip randomization.
           if (random() < self.skip_rate):
               continue
           x = y(x)
        out = self.out(x)
#        out = self.sm(out) # XXX: Verify, use softmax for probability output
        return out



# Input details.
# last_obs_value = True
# last_obs_value  = False
# RlStateUid state Uid of the attack tree node - where you are in the attack
# tree
# defender uid = 1/2

x_data = [[0, 1, 32, 1],
        [0, 1, 32, 1],
        [0, 1, 32, 1],
        [0, 1, 32, 1]]

if __name__ == "__main__":
    obs_size = 4
    n_actions = 2

    # Use GPU if have it.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu");

    my_input = torch.FloatTensor(x_data)

    nnet = Net(obs_size, HIDDEN_SIZE, n_actions).to(device)
    pred = nnet(my_input.to(device))

    print(pred)
