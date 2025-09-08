import torch
from torch import nn


class NeuralNetwork(nn.Module):
    """Multi-Layer Perceptron (MLP) neural network.

    Builds a fully connected feedforward neural network with configurable
    input, hidden, and output sizes, number of hidden layers, and activation
    function.

    Args:
        n_input (int): Number of input features.
        n_output (int): Number of output features.
        n_hidden (int): Number of units in each hidden layer.
        n_layers (int): Number of hidden layers.
        activation_function (nn.Module): Activation function to use.
        device (str): Device on which to allocate the network parameters.

    """

    def __init__(self, n_input: int, n_output: int, n_hidden: int, n_layers: int, activation_function: nn.Module, device: str) -> None:
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation_function = activation_function

        self.input_layer = nn.Sequential(
            nn.Linear(self.n_input, self.n_hidden, device=device),
            self.activation_function,
        )

        self.hidden_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden, device=device),
                self.activation_function,
            ) for _ in range(self.n_layers)],
        )

        self.output_layer = nn.Linear(self.n_hidden, self.n_output, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_input).

        Returns:
            torch.Tensor: Network output of shape (batch_size, n_output).

        """
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        return self.output_layer(x)


class InertialOscillatorLoss(nn.Module):
    """Physics-informed loss for a damped harmonic oscillator.

    Implements the motion equation for an inertial oscillator with damping
    and stiffness, penalizing deviations from the governing differential
    equation.

    Args:
        mu (torch.Tensor): Damping coefficient.
        ke (torch.Tensor): Elastic (spring) coefficient.
        m (torch.Tensor): Mass of the oscillator.

    """

    def __init__(self, mu: torch.Tensor, ke: torch.Tensor, m: torch.Tensor) -> None:
        super().__init__()

        self.mu = mu
        self.ke = ke
        self.m = m

    def forward(self, t: torch.Tensor, x_pred_entire_domain: torch.Tensor) -> torch.Tensor:
        """Compute the physics-informed loss.

        Args:
            t (torch.Tensor): Time values with gradients enabled.
            x_pred_entire_domain (torch.Tensor): Predicted displacement over the domain.

        Returns:
            torch.Tensor: Mean squared residual of the motion equation.

        """
        dx = torch.autograd.grad(x_pred_entire_domain, t, torch.ones_like(t), create_graph=True)[0]
        dx2 = torch.autograd.grad(dx,  t, torch.ones_like(dx),  create_graph=True)[0]

        diff_equation_loss = dx2 + self.mu / self.m * dx + self.ke / self.m * x_pred_entire_domain

        return torch.mean(diff_equation_loss**2)
