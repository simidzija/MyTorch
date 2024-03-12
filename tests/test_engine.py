import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import torch

from mytorch import engine
from mytorch.engine import Tensor

def test_add():

    a1 = np.random.rand(1,2,3)
    a2 = np.random.rand(2,1)
    a3 = np.random.rand(1,2,3)
    a4 = np.random.rand(4,2,3)
    a5 = np.random.rand(4,2,3)
    a6 = random.random()

    e1 = Tensor(a1, requires_grad=True)
    e2 = Tensor(a2, requires_grad=True)
    e3 = Tensor(a3)
    e4 = Tensor(a4, requires_grad=True)
    e5 = Tensor(a5)

    t1 = torch.tensor(a1, requires_grad=True)
    t2 = torch.tensor(a2, requires_grad=True)
    t3 = torch.tensor(a3)
    t4 = torch.tensor(a4, requires_grad=True)
    t5 = torch.tensor(a5)

    ex1 = e1 + e2
    ex1.backward(e3)
    tx1 = t1 + t2
    tx1.backward(t3)
    assert np.isclose(ex1.data, tx1.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    assert np.isclose(e2.grad.data, t2.grad.numpy()).all()

    ex2 = e4 + e1
    ex2.backward(e5)
    tx2 = t4 + t1
    tx2.backward(t5)
    assert np.isclose(ex2.data, tx2.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    assert np.isclose(e4.grad.data, t4.grad.numpy()).all()

    ex3 = e2 + e4
    ex3.backward(e5)
    tx3 = t2 + t4
    tx3.backward(t5)
    assert np.isclose(ex3.data, tx3.detach_().numpy()).all()
    assert np.isclose(e4.grad.data, t4.grad.numpy()).all()
    assert np.isclose(e2.grad.data, t2.grad.numpy()).all()

    ex4 = e1 + a2
    ex4.backward(e3)
    tx4 = t1 + t2
    tx4.backward(t3)
    assert np.isclose(ex4.data, tx4.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    ex5 = a6 + e1
    ex5.backward(e3)
    tx5 = a6 + t1
    tx5.backward(t3)
    assert np.isclose(ex5.data, tx5.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

def test_mul():

    a1 = np.random.rand(1,2,3)
    a2 = np.random.rand(2,1)
    a3 = np.random.rand(1,2,3)
    a4 = np.random.rand(4,2,3)
    a5 = np.random.rand(4,2,3)

    e1 = Tensor(a1, requires_grad=True)
    e2 = Tensor(a2, requires_grad=True)
    e3 = Tensor(a3)
    e4 = Tensor(a4, requires_grad=True)
    e5 = Tensor(a5)

    t1 = torch.tensor(a1, requires_grad=True)
    t2 = torch.tensor(a2, requires_grad=True)
    t3 = torch.tensor(a3)
    t4 = torch.tensor(a4, requires_grad=True)
    t5 = torch.tensor(a5)
    a6 = random.random()

    ex1 = e1 * e2
    ex1.backward(e3)
    tx1 = t1 * t2
    tx1.backward(t3)
    assert np.isclose(ex1.data, tx1.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    assert np.isclose(e2.grad.data, t2.grad.numpy()).all()

    ex2 = e4 * e1
    ex2.backward(e5)
    tx2 = t4 * t1
    tx2.backward(t5)
    assert np.isclose(ex2.data, tx2.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    assert np.isclose(e4.grad.data, t4.grad.numpy()).all()

    ex3 = e2 * e4
    ex3.backward(e5)
    tx3 = t2 * t4
    tx3.backward(t5)
    assert np.isclose(ex3.data, tx3.detach_().numpy()).all()
    assert np.isclose(e4.grad.data, t4.grad.numpy()).all()
    assert np.isclose(e2.grad.data, t2.grad.numpy()).all()

    ex4 = e1 * a2
    ex4.backward(e3)
    tx4 = t1 * t2
    tx4.backward(t3)
    assert np.isclose(ex4.data, tx4.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    ex5 = a6 * e1
    ex5.backward(e3)
    tx5 = a6 * t1
    tx5.backward(t3)
    assert np.isclose(ex5.data, tx5.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

def test_pow():

    a1 = np.random.rand(1,2,3)
    a2 = np.random.rand(2,1)
    a3 = np.random.rand(1,2,3)
    a4 = np.random.rand(4,2,3)
    a5 = np.random.rand(4,2,3)

    e1 = Tensor(a1, requires_grad=True)
    e2 = Tensor(a2, requires_grad=True)
    e3 = Tensor(a3)
    e4 = Tensor(a4, requires_grad=True)
    e5 = Tensor(a5)

    t1 = torch.tensor(a1, requires_grad=True)
    t2 = torch.tensor(a2, requires_grad=True)
    t3 = torch.tensor(a3)
    t4 = torch.tensor(a4, requires_grad=True)
    t5 = torch.tensor(a5)
    a6 = random.random()

    ex1 = e1 ** e2
    ex1.backward(e3)
    tx1 = t1 ** t2
    tx1.backward(t3)
    assert np.isclose(ex1.data, tx1.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    assert np.isclose(e2.grad.data, t2.grad.numpy()).all()

    ex2 = e4 ** e1
    ex2.backward(e5)
    tx2 = t4 ** t1
    tx2.backward(t5)
    assert np.isclose(ex2.data, tx2.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    assert np.isclose(e4.grad.data, t4.grad.numpy()).all()

    ex3 = e2 ** e4
    ex3.backward(e5)
    tx3 = t2 ** t4
    tx3.backward(t5)
    assert np.isclose(ex3.data, tx3.detach_().numpy()).all()
    assert np.isclose(e4.grad.data, t4.grad.numpy()).all()
    assert np.isclose(e2.grad.data, t2.grad.numpy()).all()

    ex4 = e1 ** a2
    ex4.backward(e3)
    tx4 = t1 ** t2
    tx4.backward(t3)
    assert np.isclose(ex4.data, tx4.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    ex5 = a6 ** e1
    ex5.backward(e3)
    tx5 = a6 ** t1
    tx5.backward(t3)
    assert np.isclose(ex5.data, tx5.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

def test_sub():

    a1 = np.random.rand(1,2,3)
    a2 = np.random.rand(2,1)
    a3 = np.random.rand(1,2,3)
    a4 = np.random.rand(4,2,3)
    a5 = np.random.rand(4,2,3)
    a6 = random.random()

    e1 = Tensor(a1, requires_grad=True)
    e2 = Tensor(a2, requires_grad=True)
    e3 = Tensor(a3)
    e4 = Tensor(a4, requires_grad=True)
    e5 = Tensor(a5)

    t1 = torch.tensor(a1, requires_grad=True)
    t2 = torch.tensor(a2, requires_grad=True)
    t3 = torch.tensor(a3)
    t4 = torch.tensor(a4, requires_grad=True)
    t5 = torch.tensor(a5)
    
    ex1 = e1 - e2
    ex1.backward(e3)
    tx1 = t1 - t2
    tx1.backward(t3)
    assert np.isclose(ex1.data, tx1.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    assert np.isclose(e2.grad.data, t2.grad.numpy()).all()

    ex2 = e4 - e1
    ex2.backward(e5)
    tx2 = t4 - t1
    tx2.backward(t5)
    assert np.isclose(ex2.data, tx2.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    assert np.isclose(e4.grad.data, t4.grad.numpy()).all()

    ex3 = e2 - e4
    ex3.backward(e5)
    tx3 = t2 - t4
    tx3.backward(t5)
    assert np.isclose(ex3.data, tx3.detach_().numpy()).all()
    assert np.isclose(e4.grad.data, t4.grad.numpy()).all()
    assert np.isclose(e2.grad.data, t2.grad.numpy()).all()

    ex4 = e1 - a2
    ex4.backward(e3)
    tx4 = t1 - t2
    tx4.backward(t3)
    assert np.isclose(ex4.data, tx4.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    ex5 = a6 - e1
    ex5.backward(e3)
    tx5 = a6 - t1
    tx5.backward(t3)
    assert np.isclose(ex5.data, tx5.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

def test_div():

    a1 = np.random.rand(1,2,3)
    a2 = np.random.rand(2,1)
    a3 = np.random.rand(1,2,3)
    a4 = np.random.rand(4,2,3)
    a5 = np.random.rand(4,2,3)
    a6 = random.random()

    e1 = Tensor(a1, requires_grad=True)
    e2 = Tensor(a2, requires_grad=True)
    e3 = Tensor(a3)
    e4 = Tensor(a4, requires_grad=True)
    e5 = Tensor(a5)

    t1 = torch.tensor(a1, requires_grad=True)
    t2 = torch.tensor(a2, requires_grad=True)
    t3 = torch.tensor(a3)
    t4 = torch.tensor(a4, requires_grad=True)
    t5 = torch.tensor(a5)
    
    ex1 = e1 / e2
    ex1.backward(e3)
    tx1 = t1 / t2
    tx1.backward(t3)
    assert np.isclose(ex1.data, tx1.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    assert np.isclose(e2.grad.data, t2.grad.numpy()).all()

    ex2 = e4 / e1
    ex2.backward(e5)
    tx2 = t4 / t1
    tx2.backward(t5)
    assert np.isclose(ex2.data, tx2.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    assert np.isclose(e4.grad.data, t4.grad.numpy()).all()

    ex3 = e2 / e4
    ex3.backward(e5)
    tx3 = t2 / t4
    tx3.backward(t5)
    assert np.isclose(ex3.data, tx3.detach_().numpy()).all()
    assert np.isclose(e4.grad.data, t4.grad.numpy()).all()
    assert np.isclose(e2.grad.data, t2.grad.numpy()).all()

    ex4 = e1 / a2
    ex4.backward(e3)
    tx4 = t1 / t2
    tx4.backward(t3)
    assert np.isclose(ex4.data, tx4.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    ex5 = a6 / e1
    ex5.backward(e3)
    tx5 = a6 / t1
    tx5.backward(t3)
    assert np.isclose(ex5.data, tx5.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

def test_matmul():

    a1 = np.random.rand(2,3)
    a2 = np.random.rand(3,4)
    a3 = np.random.rand(2,4)
    a4 = np.random.rand(5,3,4)
    a5 = np.random.rand(5,2,4)
    a6 = np.random.rand(4,2)
    a7 = np.random.rand(5,3,2)
    a8 = np.random.rand(4)
    a9 = np.random.rand(3)
    a10 = np.random.rand(5,3)
    a11 = np.random.rand(2)
    a12 = np.random.rand(2,3,4,5)
    a13 = np.random.rand(1,5,6)
    a14 = np.random.rand(2,3,4,6)

    e1 = Tensor(a1, requires_grad=True)
    e2 = Tensor(a2, requires_grad=True)
    e3 = Tensor(a3)
    e4 = Tensor(a4, requires_grad=True)
    e5 = Tensor(a5)
    e6 = Tensor(a6, requires_grad=True)
    e7 = Tensor(a7)
    e8 = Tensor(a8, requires_grad=True)
    e9 = Tensor(a9)
    e10 = Tensor(a10)
    e11 = Tensor(a11)
    e12 = Tensor(a12, requires_grad=True)
    e13 = Tensor(a13, requires_grad=True)
    e14 = Tensor(a14)

    t1 = torch.tensor(a1, requires_grad=True)
    t2 = torch.tensor(a2, requires_grad=True)
    t3 = torch.tensor(a3)
    t4 = torch.tensor(a4, requires_grad=True)
    t5 = torch.tensor(a5)
    t6 = torch.tensor(a6, requires_grad=True)
    t7 = torch.tensor(a7)
    t8 = torch.tensor(a8, requires_grad=True)
    t9 = torch.tensor(a9)
    t10 = torch.tensor(a10)
    t11 = torch.tensor(a11)
    t12 = torch.tensor(a12, requires_grad=True)
    t13 = torch.tensor(a13, requires_grad=True)
    t14 = torch.tensor(a14)

    ex1 = e1 @ e2
    ex1.backward(e3)
    tx1 = t1 @ t2
    tx1.backward(t3)
    assert np.isclose(ex1.data, tx1.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    assert np.isclose(e2.grad.data, t2.grad.numpy()).all()

    ex2 = e1 @ e4 
    ex2.backward(e5)
    tx2 = t1 @ t4 
    tx2.backward(t5)
    assert np.isclose(ex2.data, tx2.detach_().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    assert np.isclose(e4.grad.data, t4.grad.numpy()).all()

    ex3 = e4 @ e6 
    ex3.backward(e7)
    tx3 = t4 @ t6 
    tx3.backward(t7)
    assert np.isclose(ex3.data, tx3.detach_().numpy()).all()
    assert np.isclose(e4.grad.data, t4.grad.numpy()).all()
    assert np.isclose(e6.grad.data, t6.grad.numpy()).all()

    ex4 = e2 @ e8 
    ex4.backward(e9)
    tx4 = t2 @ t8 
    tx4.backward(t9)
    assert np.isclose(ex4.data, tx4.detach_().numpy()).all()
    assert np.isclose(e2.grad.data, t2.grad.numpy()).all()
    assert np.isclose(e8.grad.data, t8.grad.numpy()).all()

    ex5 = e4 @ e8 
    ex5.backward(e10)
    tx5 = t4 @ t8 
    tx5.backward(t10)
    assert np.isclose(ex5.data, tx5.detach_().numpy()).all()
    assert np.isclose(e4.grad.data, t4.grad.numpy()).all()
    assert np.isclose(e8.grad.data, t8.grad.numpy()).all()

    ex6 = e8 @ e8
    ex6.backward()
    tx6 = t8 @ t8
    tx6.backward()
    assert np.isclose(ex6.data, tx6.detach_().numpy()).all()
    assert np.isclose(e8.grad.data, t8.grad.numpy()).all()

    ex7 = e8 @ e6 
    ex7.backward(e11)
    tx7 = t8 @ t6 
    tx7.backward(t11)
    assert np.isclose(ex7.data, tx7.detach_().numpy()).all()
    assert np.isclose(e6.grad.data, t6.grad.numpy()).all()
    assert np.isclose(e8.grad.data, t8.grad.numpy()).all() 
    
    ex8 = e12 @ e13 
    ex8.backward(e14)
    tx8 = t12 @ t13 
    tx8.backward(t14)
    assert np.isclose(ex8.data, tx8.detach_().numpy()).all()
    assert np.isclose(e12.grad.data, t12.grad.numpy()).all()
    assert np.isclose(e13.grad.data, t13.grad.numpy()).all() 
    
def test_transpose():

    a1 = np.random.rand(2,3)
    a2 = np.random.rand(3,2)
    a3 = np.random.rand(2,3,4)
    a4 = np.random.rand(2,4,3)

    e1 = Tensor(a1, requires_grad=True)
    e2 = Tensor(a2)
    e3 = Tensor(a3, requires_grad=True)
    e4 = Tensor(a4)

    t1 = torch.tensor(a1, requires_grad=True)
    t2 = torch.tensor(a2)
    t3 = torch.tensor(a3, requires_grad=True)
    t4 = torch.tensor(a4)

    ex1 = e1.transpose(0,1)
    ex1.backward(e2)
    tx1 = t1.transpose(0,1)
    tx1.backward(t2)
    assert np.isclose(ex1.data, tx1.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    ex2 = e1.transpose(1,0)
    ex2.backward(e2)
    tx2 = t1.transpose(1,0)
    tx2.backward(t2)
    assert np.isclose(ex1.data, tx1.detach().numpy()).all()
 
    ex3 = e3.transpose(2,1)
    ex3.backward(e4)
    tx3 = t3.transpose(2,1)
    tx3.backward(t4)
    assert np.isclose(ex3.data, tx3.detach().numpy()).all()
    assert np.isclose(e3.grad.data, t3.grad.numpy()).all()

def test_relu():

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

    ex1 = e1.relu()
    ex1.backward(e2)
    tx1 = t1.relu()
    tx1.backward(t2)
    assert np.isclose(ex1.data, tx1.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    
    ex2 = e3.relu()
    ex2.backward(e4)
    tx2 = t3.relu()
    tx2.backward(t4)
    assert np.isclose(ex2.data, tx2.detach().numpy()).all()
    assert np.isclose(e3.grad.data, t3.grad.numpy()).all()
 
def test_getitem():

    a1 = np.random.rand(3,4,5)
    a2 = np.random.rand(*a1[1].shape)
    a3 = np.random.rand(*a1[1, 1:3:2].shape)
    a4 = np.random.rand(*a1[..., -2, :].shape)
    a5 = np.random.rand(*a1[:, None, ..., 1:3].shape)

    e1 = Tensor(a1, requires_grad=True)
    e2 = Tensor(a2)
    e3 = Tensor(a3)
    e4 = Tensor(a4)
    e5 = Tensor(a5)

    t1 = torch.tensor(a1, requires_grad=True)
    t2 = torch.tensor(a2)
    t3 = torch.tensor(a3)
    t4 = torch.tensor(a4)
    t5 = torch.tensor(a5)

    ex1 = e1[1]
    ex1.backward(e2)
    tx1 = t1[1]
    tx1.backward(t2)
    assert np.isclose(ex1.data, tx1.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    ex2 = e1[1, 1:3:2]
    ex2.backward(e3)
    tx2 = t1[1, 1:3:2]
    tx2.backward(t3)
    assert np.isclose(ex2.data, tx2.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    ex3 = e1[..., -2, :]
    ex3.backward(e4)
    tx3 = t1[..., -2, :]
    tx3.backward(t4)
    assert np.isclose(ex3.data, tx3.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    ex4 = e1[:, None, ..., 1:3]
    ex4.backward(e5)
    tx4 = t1[:, None, ..., 1:3]
    tx4.backward(t5)
    assert np.isclose(ex4.data, tx4.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

def test_stack():

    a1 = np.random.rand(4,5)
    a2 = np.random.rand(4,5)
    a3 = np.random.rand(4,5)
    a4 = np.random.rand(3,4,5)
    a5 = np.random.rand(4,3,5)

    e1 = Tensor(a1, requires_grad=True)
    e2 = Tensor(a2, requires_grad=True)
    e3 = Tensor(a3, requires_grad=True)
    e4 = Tensor(a4)
    e5 = Tensor(a5)

    t1 = torch.tensor(a1, requires_grad=True)
    t2 = torch.tensor(a2, requires_grad=True)
    t3 = torch.tensor(a3, requires_grad=True)
    t4 = torch.tensor(a4)
    t5 = torch.tensor(a5)

    ex1 = Tensor.stack([e1, e2, e3], dim=0)
    ex1.backward(e4)
    tx1 = torch.stack([t1, t2, t3], dim=0)
    tx1.backward(t4)
    assert np.isclose(ex1.data, tx1.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    assert np.isclose(e2.grad.data, t2.grad.numpy()).all()
    assert np.isclose(e3.grad.data, t3.grad.numpy()).all()

    ex2 = Tensor.stack([e1, e2, e3], dim=1)
    ex2.backward(e5)
    tx2 = torch.stack([t1, t2, t3], dim=1)
    tx2.backward(t5)
    assert np.isclose(ex2.data, tx2.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    assert np.isclose(e2.grad.data, t2.grad.numpy()).all()
    assert np.isclose(e3.grad.data, t3.grad.numpy()).all()

def test_arange():

    a1 = np.arange(3)
    a2 = np.arange(3,10)
    a3 = np.arange(3,10,2)

    e1 = engine.arange(3)
    e2 = engine.arange(3,10)
    e3 = engine.arange(3,10,2)

    assert np.isclose(a1, e1.data).all()
    assert np.isclose(a2, e2.data).all()
    assert np.isclose(a3, e3.data).all()

def test_max():

    a1 = np.random.rand(3,4,5)
    a2 = np.random.rand(4,5)
    a3 = np.random.rand(3,5)
    a4 = np.random.rand(3,4)

    e1 = Tensor(a1, requires_grad=True)
    e2 = Tensor(a2)
    e3 = Tensor(a3)
    e4 = Tensor(a4)

    t1 = torch.tensor(a1, requires_grad=True)
    t2 = torch.tensor(a2)
    t3 = torch.tensor(a3)
    t4 = torch.tensor(a4)

    ex1 = Tensor.max(e1, dim=0)
    ex1.backward(e2)
    tx1 = torch.max(t1, dim=0)[0]
    tx1.backward(t2)
    assert np.isclose(ex1.data, tx1.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    ex2 = Tensor.max(e1, dim=1)
    ex2.backward(e3)
    tx2 = torch.max(t1, dim=1)[0]
    tx2.backward(t3)
    assert np.isclose(ex2.data, tx2.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    ex3 = Tensor.max(e1, dim=2)
    ex3.backward(e4)
    tx3 = torch.max(t1, dim=2)[0]
    tx3.backward(t4)
    assert np.isclose(ex3.data, tx3.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

def test_exp():

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

    ex1 = Tensor.exp(e1)
    ex1.backward(e2)
    tx1 = torch.exp(t1)
    tx1.backward(t2)
    assert np.isclose(ex1.data, tx1.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    
    ex2 = Tensor.exp(e3)
    ex2.backward(e4)
    tx2 = torch.exp(t3)
    tx2.backward(t4)
    assert np.isclose(ex2.data, tx2.detach().numpy()).all()
    assert np.isclose(e3.grad.data, t3.grad.numpy()).all()

def test_log():

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

    ex1 = Tensor.log(e1)
    ex1.backward(e2)
    tx1 = torch.log(t1)
    tx1.backward(t2)
    assert np.isclose(ex1.data, tx1.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()
    
    ex2 = Tensor.log(e3)
    ex2.backward(e4)
    tx2 = torch.log(t3)
    tx2.backward(t4)
    assert np.isclose(ex2.data, tx2.detach().numpy()).all()
    assert np.isclose(e3.grad.data, t3.grad.numpy()).all()

def test_sum():

    a1 = np.random.rand(5)
    a2 = np.random.rand()
    a3 = np.random.rand(2,3,4)
    a4 = np.random.rand(3,4)
    a5 = np.random.rand(3)

    e1 = Tensor(a1, requires_grad=True)
    e2 = Tensor(a2)
    e3 = Tensor(a3, requires_grad=True)
    e4 = Tensor(a4)
    e5 = Tensor(a5)

    t1 = torch.tensor(a1, requires_grad=True)
    t2 = torch.tensor(a2)
    t3 = torch.tensor(a3, requires_grad=True)
    t4 = torch.tensor(a4)
    t5 = torch.tensor(a5)

    ex1 = Tensor.sum(e1, 0)
    ex1.backward(e2)
    tx1 = torch.sum(t1, 0)
    tx1.backward(t2)
    assert np.isclose(ex1.data, tx1.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    ex2 = Tensor.sum(e3, 0)
    ex2.backward(e4)
    tx2 = torch.sum(t3, 0)
    tx2.backward(t4)
    assert np.isclose(ex2.data, tx2.detach().numpy()).all()
    assert np.isclose(e3.grad.data, t3.grad.numpy()).all()

    ex3 = Tensor.sum(e3, (0,2))
    ex3.backward(e5)
    tx3 = torch.sum(t3, (0,2))
    tx3.backward(t5)
    assert np.isclose(ex3.data, tx3.detach().numpy()).all()
    assert np.isclose(e3.grad.data, t3.grad.numpy()).all()
    
def test_mean():

    a1 = np.random.rand(5)
    a2 = np.random.rand()
    a3 = np.random.rand(2,3,4)
    a4 = np.random.rand(3,4)
    a5 = np.random.rand(3)

    e1 = Tensor(a1, requires_grad=True)
    e2 = Tensor(a2)
    e3 = Tensor(a3, requires_grad=True)
    e4 = Tensor(a4)
    e5 = Tensor(a5)

    t1 = torch.tensor(a1, requires_grad=True)
    t2 = torch.tensor(a2)
    t3 = torch.tensor(a3, requires_grad=True)
    t4 = torch.tensor(a4)
    t5 = torch.tensor(a5)

    ex1 = Tensor.mean(e1, 0)
    ex1.backward(e2)
    tx1 = torch.mean(t1, 0)
    tx1.backward(t2)
    assert np.isclose(ex1.data, tx1.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    ex2 = Tensor.mean(e3, 0)
    ex2.backward(e4)
    tx2 = torch.mean(t3, 0)
    tx2.backward(t4)
    assert np.isclose(ex2.data, tx2.detach().numpy()).all()
    assert np.isclose(e3.grad.data, t3.grad.numpy()).all()

    ex3 = Tensor.mean(e3, (0,2))
    ex3.backward(e5)
    tx3 = torch.mean(t3, (0,2))
    tx3.backward(t5)
    assert np.isclose(ex3.data, tx3.detach().numpy()).all()
    assert np.isclose(e3.grad.data, t3.grad.numpy()).all()

    ex4 = Tensor.mean(e3)
    ex4.backward(e2)
    tx4 = torch.mean(t3)
    tx4.backward(t2)
    assert np.isclose(ex4.data, tx4.detach().numpy()).all()
    assert np.isclose(e3.grad.data, t3.grad.numpy()).all()

def test_cross_entropy():

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

    ex1 = Tensor.cross_entropy(e1, e3)
    ex1.backward(e5)
    tx1 = torch.nn.functional.cross_entropy(t1, t3)
    tx1.backward(t5)
    assert np.isclose(ex1.data, tx1.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    ex2 = Tensor.cross_entropy(e2, e4)
    ex2.backward(e5)
    tx2 = torch.nn.functional.cross_entropy(t2, t4)
    tx2.backward(t5)
    assert np.isclose(ex2.data, tx2.detach().numpy()).all()
    assert np.isclose(e2.grad.data, t2.grad.numpy()).all()

def test_flatten():

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

    ex1 = e1.flatten(2,3)
    ex1.backward(e2)
    tx1 = t1.flatten(2,3)
    tx1.backward(t2)
    assert np.isclose(ex1.data, tx1.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    ex2 = e1.flatten(1,-1)
    ex2.backward(e3)
    tx2 = t1.flatten(1,-1)
    tx2.backward(t3)
    assert np.isclose(ex2.data, tx2.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

    ex3 = e1.flatten()
    ex3.backward(e4)
    tx3 = t1.flatten()
    tx3.backward(t4)
    assert np.isclose(ex3.data, tx3.detach().numpy()).all()
    assert np.isclose(e1.grad.data, t1.grad.numpy()).all()

def test_argmax():

    a1 = np.random.rand(1,2,3,4)

    e1 = Tensor(a1, requires_grad=True)

    t1 = torch.tensor(a1, requires_grad=True)

    ex1 = e1.argmax()
    tx1 = t1.argmax()
    assert np.isclose(ex1.data, tx1.detach().numpy()).all()

    ex2 = e1.argmax(2)
    tx2 = t1.argmax(2)
    assert np.isclose(ex2.data, tx2.detach().numpy()).all()

    ex3 = e1.argmax(3, keepdim=True)
    tx3 = t1.argmax(3, keepdim=True)
    assert np.isclose(ex3.data, tx3.detach().numpy()).all()