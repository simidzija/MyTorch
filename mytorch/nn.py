"""
nn.py: Module for constructing neural network layers.

This module provides classes for constructing the basic building blocks of 
neural networks, including layers, parameters, and loss functions. It closely 
mirrors the structure and functionality of PyTorch's torch.nn module.

Classes:
    Module: An abstract base class for all neural network modules, providing 
        fundamental functionalities like training mode setting, parameter and 
        module registration, recursive parameter retrival, and implementing the __call__ method. 
    Parameter: Wraps a Tensor and marks it as a parameter to be optimized within
        a model.
    Linear: Implements a linear, fully connected layer.
    ReLU: Implements the rectified linear unit (ReLU) activation function.
    Sequential: A container module to chain a sequence of modules for easier
        sequential application.
    CrossEntropyLoss: Implements the cross-entropy loss function, commonly used
        for classification tasks.
    Flatten: A module that flattens multidimensional inputs along specified 
        dimensions.

Usage:
    Modules can be instantiated and combined to form a neural network. 
    
Example: 
    A simple feed-forward neural network can be constructed as follows:

        model = Sequential(
            Linear(784, 256),
            ReLU(),
            Linear(256, 10)
        )

"""

import numpy as np
from abc import ABC, abstractmethod
from collections import OrderedDict

from mytorch.engine import Tensor

class Module(ABC):
    """
    A base class for all neural network modules in MyTorch.

    This abstract class defines the standard interface and common 
    functionalities for neural network components, including setting training 
    modes, registering submodules and parameters, and defining the forward 
    pass. It is designed to be extended by specific module implementations, 
    such as layers, loss functions, and entire models.

    Attributes:
        training (bool): Indicates whether the module is in training mode.
        _modules (OrderedDict): Internal storage for submodules. 
        _parameters (OrderedDict): Internal storage for parameters.

    Methods:
        __setattr__(name, value): Automatically registers modules and
            parameters assigned as attributes.
        parameters(): Generator yielding all parameters within the module and 
            its submodules.
        train(mode=True): Sets module and all its submodules to training mode.
        eval(): Sets module and all its submodules to evaluation mode.
        __call__(*args, **kwargs): Shortcut for forward(*args, **kwargs) call.
        forward(*args, **kwargs): Abstract method defining the forward pass of 
            the module. Should be implemented by all subclasses.

    Examples:
        Defining a custom module:
            >>> class MyLayer(Module):
                    def __init__(self):
                        super().__init__()
                        self.weight = Parameter(np.random.rand(10, 5))
                        self.bias = Parameter(np.random.rand(10))

                    def forward(self, x):
                        return x @ self.weight + self.bias

            >>> layer = MyLayer()
            >>> output = layer(engine.rand(1, 5)) 

    Note:
        The Module class cannot be instantiated directly; it requires a 
        subclass that implements the forward method.
    """

    def __init__(self):
        self.training = False
        self._modules = OrderedDict()
        self._parameters = OrderedDict()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(value, Module):
            self._modules[name] = value
        if isinstance(value, Parameter):
            self._parameters[name] = value

    def parameters(self):
        for param in self._parameters.values():
            if param is not None:
                yield param

        for module in self._modules.values():
            if module is not None:
                for param in module.parameters():
                    yield param

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.training = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

class Parameter(Tensor):
    """
    A class representing a parameter in MyTorch.

    This class is a specialization of the Tensor class, intended to be used for
    model parameters that require gradient computation. Parameters are 
    typically used for weights and biases within neural network layers. By 
    default, a Parameter object has 'requires_grad' set to True, indicating 
    that gradients should be computed for model optimization.

    Inherits from:
        Tensor: The base Tensor class which allows for automatic 
        differentiation and operations on multidimensional arrays.

    Attributes:
        Inherits all attributes from the Tensor class and does not introduce any
        additional attributes. However, it implicitly sets 'requires_grad' to 
        True, marking it for gradient computation.

    Methods:
        Inherits all methods from the Tensor class.

    Examples:
        Creating a parameter for a neural network layer:
            >>> weight = Parameter(np.random.randn(10, 5))
            >>> bias = Parameter(np.random.randn(10))

    Note:
        The Parameter class is typically instantiated within the definition of 
        layer modules (subclasses of Module) where parameters are automatically 
        registered and tracked.
    """

    def __init__(self, data):
        super().__init__(data, requires_grad=True)

    def __repr__(self):
        base_repr = super().__repr__()
        return f'Parameter containing:\n' + base_repr
        

# ----------------------- Layers -----------------------

class Linear(Module):
    """
    Implements a fully connected linear layer: y = x @ A^T + b.

    Attributes:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        weight (Parameter): The weight tensor.
        bias (Parameter, None): The bias tensor. None if bias is False.

    Examples:
        Applying a linear transformation to an input batch of vectors:

            >>> layer = Linear(5, 2)
            >>> input = rand(10, 5)
            >>> output = layer(input)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        m = np.sqrt(1 / in_features)
        w_shape = (out_features, in_features)
        b_shape = (out_features)
        self.weight = Parameter(np.random.uniform(-m, m, w_shape))
        self.bias = (Parameter(np.random.uniform(-m, m, b_shape)) if bias 
                     else None)

    def forward(self, input):
        if self.bias:
            return input @ self.weight.transpose(0,1) + self.bias
        else:
            return input @ self.weight.transpose(0,1)

    def __repr__(self):
        repr = (
            f'Linear('
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'bias={bool(self.bias)})'
        )
        return repr

class ReLU(Module):
    """
    Implements the elementwise rectified linear unit (ReLU): y = relu(x)

    Examples:
        Applying a ReLU transformation to an input batch of vectors:

            >>> layer = ReLU()
            >>> input = rand(10, 5)
            >>> output = layer(input)
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.relu()
    
    def __repr__(self):
        return 'ReLU()'
    
class Sequential(Module):
    """
    A sequential container for modules. Modules will be added to it in the 
    order they are passed in the constructor. It then forwards input data 
    through each of the modules in sequence.

    Examples:
        Defining a simple feedforward neural network using Sequential:

            >>> model = Sequential(
                    Linear(784, 256),
                    ReLU(),
                    Linear(256, 10)
                )
    """

    def __init__(self, *args: Module):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module 

    def __getitem__(self, idx):
        return self._modules[str(idx)]
    
    def __len__(self):
        return len(self._modules)
    
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x
    
    def __repr__(self):
        repr = ''
        repr += f'Sequential(\n'
        for idx, module in enumerate(self._modules.values()):
            repr += f'  ({idx}): {module}\n'
        repr += ')'
        return repr

class CrossEntropyLoss(Module):
    """
    Implements the cross-entropy loss, a common loss function for 
    classification tasks. It measures the discrepancy between the predicted probabilities (input) and the true distribution (target).

    Note: the argument 'input' should be a batch of unnormalized 
    log-probabilities (logits), and the argument 'target' should be a batch of 
    class labels.

    Examples:
        Computing cross-entropy loss for some predictions and true labels:

            >>> loss_fn = CrossEntropyLoss()
            >>> predictions = Tensor([[2.0, 1.0, 0.1], [0.1, 1.0, 2.0]])
            >>> labels = Tensor([0, 2])
            >>> loss = loss_fn(predictions, labels)
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return Tensor.cross_entropy(input, target)
    
class Flatten(Module):
    """
    A module that flattens prescribed dimensions of a tensor. Useful for transforming 2D images arrays into a 1D vector form.

    Attributes:
        start_dim (int): The first dimension to flatten (default is 1).
        end_dim (int): The last dimension to flatten (default is -1).

    Examples:
        Flattening a batch of randomly generated 2D images.

            >>> images = mytorch.engine.rand(5, 28, 28)
            >>> flatten = Flatten()
            >>> flat_images = flatten(images)  # output shape will be (5, 784)
    """

    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)