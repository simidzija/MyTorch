import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import torch

from mytorch import engine, nn, optim
from mytorch.engine import Tensor


random.seed(1234)
np.random.seed(1234)

def test_SGD():

    # MyTorch network
    my_layer0 = nn.Linear(2,3)
    my_layer1 = nn.ReLU()
    my_layer2 = nn.Linear(3,5)
    my_layer3 = nn.ReLU()
    my_layer4 = nn.Linear(5,1)

    my_net = nn.Sequential(
        my_layer0,
        my_layer1,
        my_layer2,
        my_layer3,
        my_layer4
    )

    # PyTorch network with same weights and biases
    py_layer0 = torch.nn.Linear(2,3)
    py_layer1 = torch.nn.ReLU()
    py_layer2 = torch.nn.Linear(3,5)
    py_layer3 = torch.nn.ReLU()
    py_layer4 = torch.nn.Linear(5,1)

    w0 = torch.tensor(my_layer0.weight.data)
    b0 = torch.tensor(my_layer0.bias.data)
    w2 = torch.tensor(my_layer2.weight.data)
    b2 = torch.tensor(my_layer2.bias.data)
    w4 = torch.tensor(my_layer4.weight.data)
    b4 = torch.tensor(my_layer4.bias.data)

    py_layer0.weight = torch.nn.Parameter(w0, requires_grad=True)
    py_layer0.bias = torch.nn.Parameter(b0, requires_grad=True)
    py_layer2.weight = torch.nn.Parameter(w2, requires_grad=True)
    py_layer2.bias = torch.nn.Parameter(b2, requires_grad=True)
    py_layer4.weight = torch.nn.Parameter(w4, requires_grad=True)
    py_layer4.bias = torch.nn.Parameter(b4, requires_grad=True)

    py_net = torch.nn.Sequential(
        py_layer0,
        py_layer1,
        py_layer2,
        py_layer3,
        py_layer4
    )

    # Check that parameters are the same
    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.data, py_par.detach().numpy()).all()

    # SGD Optimizers with random hyperparameters
    lr = random.random()
    momentum = random.random()
    dampening = random.random()
    weight_decay = random.random()
    maximize = False

    my_optim = optim.SGD(
        my_net.parameters(), 
        lr=lr, 
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        maximize=maximize
    )

    py_optim = torch.optim.SGD(
        py_net.parameters(), 
        lr=lr, 
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        maximize=maximize
    )

    # Input tensors
    e1 = engine.rand(2, requires_grad=True)
    t1 = torch.tensor(e1.data, requires_grad=True)
    e2 = engine.rand(2, requires_grad=True)
    t2 = torch.tensor(e2.data, requires_grad=True)

    # Losses
    my_loss = my_net(e1)
    py_loss = py_net(t1)

    assert np.isclose(my_loss.data, py_loss.detach().numpy()).all()

    # Backprop and check gradients 
    my_loss.backward()
    py_loss.backward()

    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.grad.data, py_par.grad.numpy()).all()

    # Optimizer step and check updated parameters
    my_optim.step()
    py_optim.step()

    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.data, py_par.detach().numpy()).all()

    # Another step (without backprop)
    my_optim.step()
    py_optim.step()

    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.data, py_par.detach().numpy()).all()

    # Calculate loss, backprop, and step
    my_loss = my_net(e2)
    py_loss = py_net(t2)

    assert np.isclose(my_loss.data, py_loss.detach().numpy()).all()

    my_loss.backward()
    py_loss.backward()

    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.grad.data, py_par.grad.numpy()).all()

    my_optim.step()
    py_optim.step()

    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.data, py_par.detach().numpy()).all()

    # Zero grad, calculate loss, backprop, and step
    my_optim.zero_grad()
    py_optim.zero_grad()

    my_loss = my_net(e2)
    py_loss = py_net(t2)

    assert np.isclose(my_loss.data, py_loss.detach().numpy()).all()

    my_loss.backward()
    py_loss.backward()

    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.grad.data, py_par.grad.numpy()).all()

    my_optim.step()
    py_optim.step()

    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.data, py_par.detach().numpy()).all()

    # Zero grad, calculate loss, backprop, and step
    my_optim.zero_grad()
    py_optim.zero_grad()

    my_loss = my_net(e2)
    py_loss = py_net(t2)

    assert np.isclose(my_loss.data, py_loss.detach().numpy()).all()

    my_loss.backward()
    py_loss.backward()

    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.grad.data, py_par.grad.numpy()).all()

    my_optim.step()
    py_optim.step()

    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.data, py_par.detach().numpy()).all()


def test_Adam():

    # MyTorch network
    my_layer0 = nn.Linear(2,3)
    my_layer1 = nn.ReLU()
    my_layer2 = nn.Linear(3,5)
    my_layer3 = nn.ReLU()
    my_layer4 = nn.Linear(5,1)

    my_net = nn.Sequential(
        my_layer0,
        my_layer1,
        my_layer2,
        my_layer3,
        my_layer4
    )

    # PyTorch network with same weights and biases
    py_layer0 = torch.nn.Linear(2,3)
    py_layer1 = torch.nn.ReLU()
    py_layer2 = torch.nn.Linear(3,5)
    py_layer3 = torch.nn.ReLU()
    py_layer4 = torch.nn.Linear(5,1)

    w0 = torch.tensor(my_layer0.weight.data)
    b0 = torch.tensor(my_layer0.bias.data)
    w2 = torch.tensor(my_layer2.weight.data)
    b2 = torch.tensor(my_layer2.bias.data)
    w4 = torch.tensor(my_layer4.weight.data)
    b4 = torch.tensor(my_layer4.bias.data)

    py_layer0.weight = torch.nn.Parameter(w0, requires_grad=True)
    py_layer0.bias = torch.nn.Parameter(b0, requires_grad=True)
    py_layer2.weight = torch.nn.Parameter(w2, requires_grad=True)
    py_layer2.bias = torch.nn.Parameter(b2, requires_grad=True)
    py_layer4.weight = torch.nn.Parameter(w4, requires_grad=True)
    py_layer4.bias = torch.nn.Parameter(b4, requires_grad=True)

    py_net = torch.nn.Sequential(
        py_layer0,
        py_layer1,
        py_layer2,
        py_layer3,
        py_layer4
    )

    # Check that parameters are the same
    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.data, py_par.detach().numpy()).all()

    # Adam Optimizers with random hyperparameters
    lr = random.random()
    betas = (random.random(), random.random())
    eps = 1e-8
    weight_decay = random.random()
    maximize = False

    my_optim = optim.Adam(
        my_net.parameters(), 
        lr=lr, 
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        maximize=maximize
    )

    py_optim = torch.optim.Adam(
        py_net.parameters(), 
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        maximize=maximize
    )

    # Input tensors
    e1 = engine.rand(2, requires_grad=True)
    t1 = torch.tensor(e1.data, requires_grad=True)
    e2 = engine.rand(2, requires_grad=True)
    t2 = torch.tensor(e2.data, requires_grad=True)

    # Losses
    my_loss = my_net(e1)
    py_loss = py_net(t1)

    assert np.isclose(my_loss.data, py_loss.detach().numpy()).all()

    # Backprop and check gradients 
    my_loss.backward()
    py_loss.backward()

    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.grad.data, py_par.grad.numpy()).all()

    # Optimizer step and check updated parameters
    my_optim.step()
    py_optim.step()

    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.data, py_par.detach().numpy()).all()

    # Another step (without backprop)
    my_optim.step()
    py_optim.step()

    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.data, py_par.detach().numpy()).all()

    # Calculate loss, backprop, and step
    my_loss = my_net(e2)
    py_loss = py_net(t2)

    assert np.isclose(my_loss.data, py_loss.detach().numpy()).all()

    my_loss.backward()
    py_loss.backward()

    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.grad.data, py_par.grad.numpy()).all()

    my_optim.step()
    py_optim.step()

    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.data, py_par.detach().numpy()).all()

    # Zero grad, calculate loss, backprop, and step
    my_optim.zero_grad()
    py_optim.zero_grad()

    my_loss = my_net(e2)
    py_loss = py_net(t2)

    assert np.isclose(my_loss.data, py_loss.detach().numpy()).all()

    my_loss.backward()
    py_loss.backward()

    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.grad.data, py_par.grad.numpy()).all()

    my_optim.step()
    py_optim.step()

    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.data, py_par.detach().numpy()).all()

    # Zero grad, calculate loss, backprop, and step
    my_optim.zero_grad()
    py_optim.zero_grad()

    my_loss = my_net(e2)
    py_loss = py_net(t2)

    assert np.isclose(my_loss.data, py_loss.detach().numpy()).all()

    my_loss.backward()
    py_loss.backward()

    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.grad.data, py_par.grad.numpy()).all()

    my_optim.step()
    py_optim.step()

    for my_par, py_par in zip(my_net.parameters(), py_net.parameters()):
        assert np.isclose(my_par.data, py_par.detach().numpy()).all()