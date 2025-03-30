"""
This script demonstrates the implementation of a spiking convolutional classifier 
using the Norse library for spiking neural networks. The model combines traditional 
convolutional and pooling layers with spiking neuron layers to process input data.

Modules:
    - torch: PyTorch library for tensor computations and neural networks.
    - norse.torch: Norse library for spiking neural network components.

Classes and Functions:
    - SequentialState: A sequential container for layers with stateful processing.
    - LICell: A leaky integrator cell for non-spiking integration.
    - LIFCell: A leaky integrate-and-fire cell for spiking activation.

Model Architecture:
    - Convolutional layers with 5x5 kernels and increasing channels (1 -> 20 -> 50).
    - Spiking activation layers (LIFCell) after each convolutional layer.
    - Max pooling layers to reduce spatial dimensions.
    - Fully connected layer with 800 input units and 10 output units.
    - Final non-spiking integrator layer (LICell).

Example Usage:
    - Random input data with shape (8, 1, 28, 28) is passed through the model.
    - The model outputs a tuple containing:
        - A tensor of shape (8, 10) representing the output for 8 timesteps.
        - The internal state of the neurons.

Note:
    - This example is adapted from the Norse documentation and serves as a 
      demonstration of spiking neural network capabilities.
"""

import torch, torch.nn as nn
from norse.torch import LICell             # Leaky integrator
from norse.torch import LIFCell            # Leaky integrate-and-fire
from norse.torch import SequentialState    # Stateful sequential layers

model = SequentialState(
    nn.Conv2d(1, 20, 5, 1),      # Convolve from 1 -> 20 channels
    LIFCell(),                   # Spiking activation layer
    nn.MaxPool2d(2, 2),
    nn.Conv2d(20, 50, 5, 1),     # Convolve from 20 -> 50 channels
    LIFCell(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),                # Flatten to 800 units
    nn.Linear(800, 10),
    LICell(),                    # Non-spiking integrator layer
)

# Random data to run model
data = torch.randn(8, 1, 28, 28) # 8 timesteps, 1 channel, 28x28 pixels
output, state = model(data)      # Provides a tuple (tensor (8, 10), neuron state)
