import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_layer_sizes,
                 output_size,
                 dropout_rate=0.5):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_layer_sizes
        self.output_size = output_size

        self.l0 = nn.Linear(input_size, hidden_layer_sizes[0])

        self.l1 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])

        # self.l2 = nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2])

        # Insert dropout layer here
        self.dropout = nn.Dropout(dropout_rate)

        self.l3 = nn.Linear(hidden_layer_sizes[-1], output_size)

        self.relu = nn.ReLU()

        self.tanh = nn.Tanh()

    def forward(self, x):
        
        out = self.l0(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.l1(out)
        out = self.relu(out)
        out = self.dropout(out)

        # out = self.l2(out)
        # out = self.relu(out)
        # out = self.dropout(out)

        out = self.l3(out)

        return out
