import torch.nn as nn

Activations={
    "ReLU":nn.ReLU(inplace=True)
}

class FC(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0, activation=None):
        super(FC, self).__init__()

        self.hasactivation = activation is not None

        self.linear = nn.Linear(input_dim, output_dim)

        if activation is not None:
            self.activation = Activations[activation]

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout=None

    def forward(self, x):
        x = self.linear(x)

        if self.hasactivation:
            x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0, activation=None):
        super(MLP, self).__init__()

        self.fc = FC(input_dim, hidden_dim, dropout=dropout, activation=activation)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear(self.fc(x))
