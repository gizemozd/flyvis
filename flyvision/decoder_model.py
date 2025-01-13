""" Module containing the decoder model. """
import torch
from torch import nn

activation_fns = {
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "softplus": nn.Softplus(),
    "leakyrelu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "hardtanh": nn.Hardtanh(min_val=0.0, max_val=5.0),
}


class SimpleRNNDecoder(nn.Module):
    """Simple RNN decoder module.
    For now, the best params I could find were:
    hidden_size = 7 * 4 = 28
    n_layers = 2
    nonlinearity = tanh looks better but more noisy?
    relu prevents vanishing gradients?
    """

    def __init__(self, input_size, hidden_size, n_layers, output_size, **kwargs):
        super(SimpleRNNDecoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        nonlinearity = kwargs.get("nonlinearity", "tanh")
        dropout = kwargs.get("dropout", 0.2)

        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            n_layers,
            nonlinearity=nonlinearity,
            dropout=dropout,
            batch_first=True,
            bidirectional=False,
        )
        self.fully_connected = nn.Linear(hidden_size, output_size)

    def forward(self, x, initial_state=None):
        """Forward pass through the network."""
        batch_size = x.size(0)
        # Initializing hidden state for first input using method defined below
        if initial_state is None:
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        else:
            hidden = initial_state

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Fully connected layer
        out = self.fully_connected(out)

        return out, hidden


class SimpleDecoder(nn.Module):
    def __init__(
            self, 
            activation_function: str, 
            in_features: int,
            hidden_features: int, 
            out_features: int, 
            dropout: float = 0.2
        ):
        super(SimpleDecoder, self).__init__()
        self.act_function = activation_fns[activation_function]
        # First layer
        self.first_layer = nn.Linear(
            in_features=in_features, out_features=hidden_features, bias=True
        )
        self.second_layer = nn.Linear(
            in_features=hidden_features, out_features=out_features, bias=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, 6)
        x = self.first_layer(x)
        x = self.act_function(x)
        x = self.dropout(x)
        x = self.second_layer(x)
        return x


class SimpleDecoderTwin(nn.Module):
    def __init__(self, activation_function: str, in_features: int, out_features: int):
        super(SimpleDecoderTwin, self).__init__()
        self.act_function = activation_fns[activation_function]
        self.linear_first = nn.Linear(
            in_features=in_features, out_features=out_features, bias=True
        )  # Learnable weights for the weighted sum
        self.linear_second = nn.Linear(
            in_features=in_features, out_features=out_features, bias=True
        )  # Learnable weights for the weighted sum

    def forward(self, x_first, x_second):
        # x shape: (batch_size, sequence_length, 6)
        sigmoid_outputs_first = self.act_function(x_first)  # Apply sigmoid activation to each time step
        sigmoid_outputs_second = self.act_function(x_second)  # Apply sigmoid activation to each time step

        return (self.linear_first(sigmoid_outputs_first), self.linear_second(sigmoid_outputs_second))
