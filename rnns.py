"""Simple recurrent neural network architectures for time-series forecasting.

This module implements three shallow recurrent architectures using PyTorch:

1. ElmanRNN
   The hidden layer receives feedback from the previous hidden state.

2. JordanRNN
   The hidden layer receives feedback from the previous network output.

3. MultiRecurrentRNN
   The hidden layer receives both the previous hidden state and the previous
   network output.

All models share the same external interface:

    model = Model(input_size=1, hidden_size=32, output_size=1)
    y_hat = model(x)

where:

    x.shape == (batch_size, sequence_length, input_size)
    y_hat.shape == (batch_size, output_size)

The recurrence is implemented explicitly with nn.Linear layers rather than
using nn.RNN so that the three architectures differ only in their recurrent
connections while sharing the same overall implementation structure.
"""


from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseSimpleRNN(nn.Module, ABC):
    """Base class shared by the three recurrent forecasting models."""

    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        output_size=1,
        activation="tanh",
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_name = activation

        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=True)
        self.hidden_to_output = nn.Linear(hidden_size, output_size, bias=True)
        self.activation = self._make_activation(activation)

    @staticmethod
    def _make_activation(name):
        """Create the hidden activation function."""
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
        }

        return activations[name]

    def reset_parameters(self):
        """Initialize all linear layers using Xavier uniform weights and zero biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_input(self, x):
        """Validate the expected batch-first sequence input shape."""
        if x.ndim != 3:
            raise ValueError(
                "Expected x with shape "
                "(batch_size, sequence_length, input_size)"
            )

        if x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected input_size={self.input_size}, "
                f"but received {x.size(-1)} features."
            )

        if x.size(1) == 0:
            raise ValueError("sequence_length must be at least 1")

    def _zeros(
        self,
        batch_size,
        width,
        reference,
    ):
        """Create a zero state matching the input tensor's dtype and device."""
        return reference.new_zeros(batch_size, width)

    @abstractmethod
    def forward(
        self,
        x,
        return_sequence=False,
    ):
        """Run the model over a complete input sequence."""


class ElmanRNN(BaseSimpleRNN):
    """Elman recurrent neural network.

    Recurrence:
        h_t = f(W_x x_t + W_h h_(t-1) + b_h)
        y_t = W_o h_t + b_o
    """

    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        output_size=1,
        activation="tanh",
    ):
        super().__init__(input_size, hidden_size, output_size, activation)
        self.hidden_to_hidden = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )
        self.reset_parameters()

    def forward(
        self,
        x,
        return_sequence=False,
    ):
        """Process a batch of sequences."""
        self._validate_input(x)

        batch_size, sequence_length, _ = x.shape
        hidden = self._zeros(batch_size, self.hidden_size, x)
        outputs = []

        for t in range(sequence_length):
            x_t = x[:, t, :]

            hidden = self.activation(
                self.input_to_hidden(x_t)
                + self.hidden_to_hidden(hidden)
            )

            output = self.hidden_to_output(hidden)

            if return_sequence:
                outputs.append(output)

        if return_sequence:
            return torch.stack(outputs, dim=1)

        return output


class JordanRNN(BaseSimpleRNN):
    """Jordan recurrent neural network.

    Recurrence:
        h_t = f(W_x x_t + W_y y_(t-1) + b_h)
        y_t = W_o h_t + b_o
    """

    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        output_size=1,
        activation="tanh",
    ):
        super().__init__(input_size, hidden_size, output_size, activation)
        self.output_to_hidden = nn.Linear(
            output_size,
            hidden_size,
            bias=False,
        )
        self.reset_parameters()

    def forward(
        self,
        x,
        return_sequence=False,
    ):
        """Process a batch of sequences."""
        self._validate_input(x)

        batch_size, sequence_length, _ = x.shape
        previous_output = self._zeros(batch_size, self.output_size, x)
        outputs = []

        for t in range(sequence_length):
            x_t = x[:, t, :]

            hidden = self.activation(
                self.input_to_hidden(x_t)
                + self.output_to_hidden(previous_output)
            )

            output = self.hidden_to_output(hidden)
            previous_output = output

            if return_sequence:
                outputs.append(output)

        if return_sequence:
            return torch.stack(outputs, dim=1)

        return output


class MultiRecurrentRNN(BaseSimpleRNN):
    """Multi-recurrent neural network.

    This follows the course architecture where both the previous hidden state
    and the previous network output feed back into the hidden layer.

    Recurrence:
        h_t = f(
            W_x x_t
            + W_h h_(t-1)
            + W_y y_(t-1)
            + b_h
        )
        y_t = W_o h_t + b_o
    """

    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        output_size=1,
        activation="tanh",
    ):
        super().__init__(input_size, hidden_size, output_size, activation)

        self.hidden_to_hidden = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )
        self.output_to_hidden = nn.Linear(
            output_size,
            hidden_size,
            bias=False,
        )
        self.reset_parameters()

    def forward(
        self,
        x,
        return_sequence=False,
    ):
        """Process a batch of sequences."""
        self._validate_input(x)

        batch_size, sequence_length, _ = x.shape
        hidden = self._zeros(batch_size, self.hidden_size, x)
        previous_output = self._zeros(batch_size, self.output_size, x)
        outputs = []

        for t in range(sequence_length):
            x_t = x[:, t, :]

            hidden = self.activation(
                self.input_to_hidden(x_t)
                + self.hidden_to_hidden(hidden)
                + self.output_to_hidden(previous_output)
            )

            output = self.hidden_to_output(hidden)
            previous_output = output

            if return_sequence:
                outputs.append(output)

        if return_sequence:
            return torch.stack(outputs, dim=1)

        return output


MODEL_REGISTRY = {
    "elman": ElmanRNN,
    "jordan": JordanRNN,
    "multi": MultiRecurrentRNN,
}


def build_rnn(
    architecture,
    input_size=1,
    hidden_size=32,
    output_size=1,
    activation="tanh",
):
    """Construct one of the supported recurrent architectures."""
    key = architecture.lower()


    return MODEL_REGISTRY[key](
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        activation=activation,
    )
