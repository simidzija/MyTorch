"""
optim.py: Module for MyTorch optimizers.

This module implements optimization algorithms used in training neural 
networks. These optimizers adjust the parameters of the network with respect to 
the gradients of a loss function during training in an attempt to minimize 
loss. 

Classes:
    Optimizer: An abstract base class for all optimizers, defining common 
        functionalities such as parameter groups, a state dict, and a method for zeroing gradients.
    SGD: Implements the Stochastic Gradient Descent optimization algorithm,
        including options for momentum, weight decay, and dampening.

Usage:
    Optimizers are typically used in conjunction with neural network models to 
    update model parameters during training based on the computed gradients.

Example:
    Setting up and using an SGD optimizer with a neural network model:

        from mytorch.optim import SGD
        from mytorch.nn import Sequential, Linear, ReLU, CrossEntropyLoss

        model = Sequential(
            Linear(784, 256),
            ReLU(),
            Linear(256, 10)
        )
        criterion = CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Training loop
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

"""

from abc import ABC, abstractmethod
from collections import defaultdict

import mytorch.engine as engine
from mytorch.engine import Tensor

class Optimizer(ABC):
    """
    A base class for all optimization algorithms in MyTorch.

    This abstract class provides the structure and common functionalities 
    required for implementing various optimization algorithms used to update 
    neural network parameters based on their gradients. 

    Attributes:
        param_groups (list): A list of parameter groups, each of which is a 
            dictionary that holds parameters and their corresponding 
            optimization settings.
        defaults (dict): Default optimization settings for all parameter groups.
        state (defaultdict): A defaultdict storing parameters as keys and dicts 
            containing state dependent optimization settings for the given 
            parameter as values.

    Methods:
        step(): Abstract method that updates the parameters based on their 
            gradients. Must be implemented by subclasses.
        zero_grad(): Clears the gradients of all parameters in this optimizer, 
            preparing them for the next forward and backward passes.

    Examples:
        Creating a custom optimizer:
            >>> class MyOptimizer(Optimizer):
                    def __init__(self, params, lr):
                        defaults = {
                            'lr': lr
                        }
                        super().__init__(params, defaults)

                    def step(self):
                        # Update parameters based on custom optimization logic

            >>> optimizer = MyOptimizer(model.parameters(), lr=0.01)

    Note:
        The Optimizer class is designed to be subclassed with specific implementations of the step() method.
    """

    def __init__(self, params, defaults):
        self.param_groups = [{}]
        self.param_groups[0]['params'] = [param for param in params]
        for def_param, value in defaults.items():
            self.param_groups[0][def_param] = value

        self.defaults = defaults

        self.state = defaultdict(dict)

    def __repr__(self):
        repr = 'Optimizer (\n'
        for idx, group_dict in enumerate(self.param_groups):
            repr += f'Parameter Group {idx}\n'
            for param, value in group_dict.items():
                if param == 'params':
                    pass
                else:
                    repr += f'    {param}: {value}\n'
        repr += ')'
        return repr

    @abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        for group in self.param_groups:
            for param in group['params']:
                if param is not None:
                    param.grad.zero_()


class SGD(Optimizer):
    """
    Implements the Stochastic Gradient Descent (SGD) optimization algorithm.

    SGD updates parameters by moving in the direction of the negative gradient, 
    optionally with momentum, dampening, and weight decay.

    Inherits from:
        Optimizer: The base Optimizer class in MyTorch which provides common 
            functionalities for all optimizers.

    Attributes:
        Inherits all attributes from the Optimizer class. Additionally, defines 
        the defaults dict containing default optimization parameters, including
        learning rate (lr), momentum (mu), dampening (tau), weight decay 
        (lambd), and a maximization toggle (maximize).

    Methods:
        step(): Implements the SGD parameter update method.

    Examples:
        Using the SGD optimizer with a simple neural network model:
            >>> from mytorch.nn import Sequential, Linear, ReLU
            >>> from mytorch.optim import SGD

            >>> model = Sequential(
                    Linear(784, 256),
                    ReLU(),
                    Linear(256, 10)
                )
            >>> criterion = CrossEntropyLoss()
            >>> optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

            >>> # Training loop
            >>> for inputs, targets in dataloader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

    """

    def __init__(self, params, lr, momentum=0, dampening=0,
                  weight_decay=0, *, maximize=False):

        defaults = {
            'lr': lr,
            'momentum': momentum,
            'dampening': dampening,
            'weight_decay': weight_decay,
            'maximize': maximize
        }
        super().__init__(params, defaults)

    def step(self):
        for param_group in self.param_groups:
            lr = param_group['lr']
            mu = param_group['momentum']
            tau = param_group['dampening']
            lambd = param_group['weight_decay']
            maximize = param_group['maximize']

            for p in param_group['params']:
                # copy p's gradient to avoid changing it via in-place operations
                g = p.grad.data.copy()
                if lambd != 0:
                    g += lambd * p.data
                if mu != 0:
                    if 'momentum_buffer' in self.state[p]:
                        self.state[p]['momentum_buffer'].data *= mu 
                        self.state[p]['momentum_buffer'].data += (1 - tau) * g 
                    else:
                        self.state[p]['momentum_buffer'] = Tensor(g)
                    g = self.state[p]['momentum_buffer'].data
                # update p.data rather than p, so that p's ID remains unchanged
                if maximize:
                    p.data += lr * g
                else:
                    p.data -= lr * g

class Adam(Optimizer):

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, *, maximize=False):
        
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'maximize': maximize
        }

        super().__init__(params, defaults)

    def step(self):
        for param_group in self.param_groups:
            lr = param_group['lr']
            beta1, beta2 = param_group['betas']
            eps = param_group['eps']
            lambd = param_group['weight_decay']
            maximize = param_group['maximize']

            for p in param_group['params']:
                # copy p's gradient to avoid changing it via in-place operations
                g = (-1 if maximize else +1) * p.grad.data.copy()
                # weight decay
                if lambd != 0:
                    g += lambd * p.data
                # step, biased first moment, biased second moment
                if 'step' not in self.state[p]:
                    self.state[p]['step'] = 0
                    self.state[p]['exp_avg'] = engine.zeros_like(g)
                    self.state[p]['exp_avg_sq'] = engine.zeros_like(g)
                self.state[p]['step'] += 1
                self.state[p]['exp_avg'] = (
                    beta1 * self.state[p]['exp_avg'] + 
                    (1 - beta1) * g
                )
                self.state[p]['exp_avg_sq'] = (
                    beta2 * self.state[p]['exp_avg_sq'] + 
                    (1 - beta2) * g**2
                )
                t = self.state[p]['step']
                m = self.state[p]['exp_avg']
                v = self.state[p]['exp_avg_sq']
                # unbiased first and second moments
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                # update p.data rather than p, so that p's ID remains unchanged
                p.data -= lr * m_hat.data / (v_hat.data**0.5 + eps)
