import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from mytorch import nn
from mytorch.engine import Tensor


def test_Linear():

    my_layer0 = nn.Linear(2, 3)
    py_layer0 = torch.nn.Linear(2, 3)
    my_layer1 = nn.Linear(2, 3, bias=False)
    py_layer1 = torch.nn.Linear(2, 3, bias=False)

    # set torch layers' weights and biases equal to those of mytorch layers
    weight0 = torch.tensor(my_layer0.weight.data)
    bias0 = torch.tensor(my_layer0.bias.data)
    py_layer0.weight = torch.nn.Parameter(weight0, requires_grad=True)
    py_layer0.bias = torch.nn.Parameter(bias0, requires_grad=True)

    weight2 = torch.tensor(my_layer1.weight.data)
    py_layer1.weight = torch.nn.Parameter(weight2, requires_grad=True)

    a1 = np.random.rand(2)
    a2 = np.random.rand(3)
    a3 = np.random.rand(5,2) # batch input
    a4 = np.random.rand(5,3)

    e1 = Tensor(a1, requires_grad=True)
    e2 = Tensor(a2)
    e3 = Tensor(a3, requires_grad=True)
    e4 = Tensor(a4)

    t1 = torch.tensor(a1, requires_grad=True)
    t2 = torch.tensor(a2)
    t3 = torch.tensor(a3, requires_grad=True)
    t4 = torch.tensor(a4)

    ex1 = my_layer0(e1)
    ex1.backward(e2)
    tx1 = py_layer0(t1)
    tx1.backward(t2)
    assert np.isclose(ex1.data, tx1.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    assert np.isclose(my_layer0.weight.grad.data, 
                    py_layer0.weight.grad.numpy()).all()
    assert np.isclose(my_layer0.bias.grad.data, 
                    py_layer0.bias.grad.numpy()).all()

    ex2 = my_layer0(e3)
    ex2.backward(e4)
    tx2 = py_layer0(t3)
    tx2.backward(t4)
    assert np.isclose(ex2.data, tx2.detach().numpy()).all()
    assert np.isclose(e3.grad.data, t3.grad.numpy()).all()
    assert np.isclose(my_layer0.weight.grad.data, 
                    py_layer0.weight.grad.numpy()).all()
    assert np.isclose(my_layer0.bias.grad.data, 
                    py_layer0.bias.grad.numpy()).all()
    
    ex3 = my_layer1(e1)
    ex3.backward(e2)
    tx3 = py_layer1(t1)
    tx3.backward(t2)
    assert np.isclose(ex3.data, tx3.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    assert np.isclose(my_layer1.weight.grad.data, 
                    py_layer1.weight.grad.numpy()).all()

    ex4 = my_layer1(e3)
    ex4.backward(e4)
    tx4 = py_layer1(t3)
    tx4.backward(t4)
    assert np.isclose(ex4.data, tx4.detach().numpy()).all()
    assert np.isclose(e3.grad.data, t3.grad.numpy()).all()
    assert np.isclose(my_layer1.weight.grad.data, 
                    py_layer1.weight.grad.numpy()).all()
    

def test_ReLU():

    my_layer = nn.ReLU()
    py_layer = torch.nn.ReLU()

    a1 = np.random.rand(2)
    a2 = np.random.rand(2)
    a3 = np.random.rand(2,3,4)
    a4 = np.random.rand(2,3,4)

    e1 = Tensor(a1, requires_grad=True)
    e2 = Tensor(a2)
    e3 = Tensor(a3, requires_grad=True)
    e4 = Tensor(a4)

    t1 = torch.tensor(a1, requires_grad=True)
    t2 = torch.tensor(a2)
    t3 = torch.tensor(a3, requires_grad=True)
    t4 = torch.tensor(a4)

    ex1 = my_layer(e1)
    ex1.backward(e2)
    tx1 = py_layer(t1)
    tx1.backward(t2)
    assert np.isclose(ex1.data, tx1.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    
    ex2 = my_layer(e3)
    ex2.backward(e4)
    tx2 = py_layer(t3)
    tx2.backward(t4)
    assert np.isclose(ex2.data, tx2.detach().numpy()).all()
    assert np.isclose(e3.grad.data, t3.grad.numpy()).all()


def test_Sequential():

    my_layer0 = nn.Linear(2,3)
    my_layer1 = nn.ReLU()
    my_layer2 = nn.Linear(3,4)
    my_layer3 = nn.ReLU()
    my_layer4 = nn.Linear(4,5)

    py_layer0 = torch.nn.Linear(2,3)
    py_layer1 = torch.nn.ReLU()
    py_layer2 = torch.nn.Linear(3,4)
    py_layer3 = torch.nn.ReLU()
    py_layer4 = torch.nn.Linear(4,5)
    
    # set torch layers' weights and biases equal to those of mytorch layers
    weight0 = torch.tensor(my_layer0.weight.data)
    weight2 = torch.tensor(my_layer2.weight.data)
    weight4 = torch.tensor(my_layer4.weight.data)
    bias0 = torch.tensor(my_layer0.bias.data)
    bias2 = torch.tensor(my_layer2.bias.data)
    bias4 = torch.tensor(my_layer4.bias.data)
    py_layer0.weight = torch.nn.Parameter(weight0, requires_grad=True)
    py_layer2.weight = torch.nn.Parameter(weight2, requires_grad=True)
    py_layer4.weight = torch.nn.Parameter(weight4, requires_grad=True)
    py_layer0.bias = torch.nn.Parameter(bias0, requires_grad=True)
    py_layer2.bias = torch.nn.Parameter(bias2, requires_grad=True)
    py_layer4.bias = torch.nn.Parameter(bias4, requires_grad=True)

    my_sequential = nn.Sequential(
        my_layer0,
        my_layer1,
        my_layer2,
        my_layer3,
        my_layer4
    )

    py_sequential = torch.nn.Sequential(
        py_layer0,
        py_layer1,
        py_layer2,
        py_layer3,
        py_layer4
    )

    a1 = np.random.rand(2)      # input
    a2 = np.random.rand(5)      # output grad
    a3 = np.random.rand(10,2)   # batch input
    a4 = np.random.rand(10,5)   # batch output grad

    e1 = Tensor(a1, requires_grad=True)
    e2 = Tensor(a2)
    e3 = Tensor(a3, requires_grad=True)
    e4 = Tensor(a4)

    t1 = torch.tensor(a1, requires_grad=True)
    t2 = torch.tensor(a2)
    t3 = torch.tensor(a3, requires_grad=True)
    t4 = torch.tensor(a4)

    # single input
    ex1 = my_sequential(e1)
    ex1.backward(e2)
    tx1 = py_sequential(t1)
    tx1.backward(t2)
    # output
    assert np.isclose(ex1.data, tx1.detach().numpy()).all()
    # input grad
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    # layer0 grads
    assert np.isclose(my_layer0.weight.grad.data, 
                    py_layer0.weight.grad.numpy()).all()
    assert np.isclose(my_layer0.bias.grad.data, 
                    py_layer0.bias.grad.numpy()).all()
    # layer2 grads
    assert np.isclose(my_layer2.weight.grad.data, 
                    py_layer2.weight.grad.numpy()).all()
    assert np.isclose(my_layer2.bias.grad.data, 
                    py_layer2.bias.grad.numpy()).all()
    # layer4 grads
    assert np.isclose(my_layer4.weight.grad.data, 
                    py_layer4.weight.grad.numpy()).all()
    assert np.isclose(my_layer4.bias.grad.data, 
                    py_layer4.bias.grad.numpy()).all()

    # batch input
    ex2 = my_sequential(e3)
    ex2.backward(e4)
    tx2 = py_sequential(t3)
    tx2.backward(t4)
    # output
    assert np.isclose(ex2.data, tx2.detach().numpy()).all()
    # input grad
    assert np.isclose(e3.grad.data, t3.grad.numpy()).all()
    # layer0 grads
    assert np.isclose(my_layer0.weight.grad.data, 
                    py_layer0.weight.grad.numpy()).all()
    assert np.isclose(my_layer0.bias.grad.data, 
                    py_layer0.bias.grad.numpy()).all()
    # layer2 grads
    assert np.isclose(my_layer2.weight.grad.data, 
                    py_layer2.weight.grad.numpy()).all()
    assert np.isclose(my_layer2.bias.grad.data, 
                    py_layer2.bias.grad.numpy()).all()
    # layer4 grads
    assert np.isclose(my_layer4.weight.grad.data, 
                    py_layer4.weight.grad.numpy()).all()
    assert np.isclose(my_layer4.bias.grad.data, 
                    py_layer4.bias.grad.numpy()).all()

def test_CrossEntropyLoss():

    my_loss = nn.CrossEntropyLoss()
    py_loss = torch.nn.CrossEntropyLoss()

    a1 = np.random.rand(1,5)              # input: batch size 1
    a2 = np.random.rand(3,5)              # input: batch size 3 
    a3 = np.random.randint(5, size=(1,))  # target: batch size 1
    a4 = np.random.randint(5, size=(3,))  # target: batch size 3
    a5 = np.random.rand()                 # grad_output
    
    e1 = Tensor(a1, requires_grad=True)
    e2 = Tensor(a2, requires_grad=True)
    e3 = Tensor(a3)
    e4 = Tensor(a4)
    e5 = Tensor(a5)

    t1 = torch.tensor(a1, requires_grad=True)
    t2 = torch.tensor(a2, requires_grad=True)
    t3 = torch.tensor(a3)
    t4 = torch.tensor(a4)
    t5 = torch.tensor(a5)

    ex1 = my_loss(e1, e3)
    ex1.backward(e5)
    tx1 = py_loss(t1, t3)
    tx1.backward(t5)
    assert np.isclose(ex1.data, tx1.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    ex2 = my_loss(e2, e4)
    ex2.backward(e5)
    tx2 = py_loss(t2, t4)
    tx2.backward(t5)
    assert np.isclose(ex2.data, tx2.detach().numpy()).all()
    assert np.isclose(e2.grad.data, t2.grad.numpy()).all()

def test_Flatten():

    a1 = np.random.rand(1,2,3,4)     # input
    a2 = np.random.rand(1,2,12)      # grad: flatten dims 2-3
    a3 = np.random.rand(1,24)        # grad: flatten dims 1-3
    a4 = np.random.rand(24)          # grad: flatten dims 0-3

    e1 = Tensor(a1, requires_grad=True)
    e2 = Tensor(a2)
    e3 = Tensor(a3)
    e4 = Tensor(a4)

    t1 = torch.tensor(a1, requires_grad=True)
    t2 = torch.tensor(a2)
    t3 = torch.tensor(a3)
    t4 = torch.tensor(a4)

    my_layer = nn.Flatten(2,3)
    py_layer = torch.nn.Flatten(2,3)
    ex1 = my_layer(e1)
    ex1.backward(e2)
    tx1 = py_layer(t1)
    tx1.backward(t2)
    assert np.isclose(ex1.data, tx1.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    my_layer = nn.Flatten()
    py_layer = torch.nn.Flatten()
    ex2 = my_layer(e1)
    ex2.backward(e3)
    tx2 = py_layer(t1)
    tx2.backward(t3)
    assert np.isclose(ex2.data, tx2.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    my_layer = nn.Flatten(0)
    py_layer = torch.nn.Flatten(0)
    ex3 = my_layer(e1)
    ex3.backward(e4)
    tx3 = py_layer(t1)
    tx3.backward(t4)
    assert np.isclose(ex3.data, tx3.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()