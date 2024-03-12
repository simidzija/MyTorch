"""
engine.py: A module for defining and manipulating tensors.

This module contains the Tensor class, implementing basic tensor operations and 
the backpropagation algorithm.

Classes:
    Tensor: A class that represents a multi-dimensional array and includes
        functionality for automatic reverse-mode differentiation.

Methods:
    zeros, ones, rand, zeros_like, ones_like, rand_like, and arange: A set of 
    functions for creating tensors of specified shape and containing zeros, 
    ones, or random values.

Usage:
    This module, and specifically the Tensor class, form the foundation of the 
    MyTorch library, and is intended to be used in neural network constructions 
    where gradient computation via backpropagation is necessary.

Example:
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = rand([2, 2], requires_grad=True)
    z = x + y
    z.backward()
"""

import numpy as np

class Tensor:
    """
    A class for MyTorch Tensors, allowing for automatic differentiation and 
    operations on multidimensional arrays.

    Supports basic tensor operations and enables automatic differentiation by
    tracking computational history when 'requires_grad' is True. Intended for
    gradient-based optimization tasks in neural networks. 

    Attributes:
        data (numpy.ndarray): Raw numerical data in the tensor.
        requires_grad (bool): Whether gradient of some loss function with 
            respect to Tensor will be required.
        shape (tuple): Dimensions of the tensor.
        ndim (int): Number of dimensions of the tensor.
        grad (Tensor or None): Gradient from backpropagation, same shape as this
            tensor, None if 'requires_grad' is False.
        _prev (set): Internal; predecessors in computational graph.

    Methods:
        requires_grad_(mode=True): Enable/disable gradient tracking.
        numel(): Returns the total number of elements.
        backward(gradient=None): Initiates backward pass for gradient 
            computation.

    Basic Operations Supported:
        Supports various operations compatible with backpropagation, including 
        addition (__add__), subtraction (__sub__), multiplication (__mul__), 
        division (__truediv__), power (__pow__), matrix multiplication 
        (__matmul__), slicing (__getitem__), elementwise ReLU (relu), 
        transposition (transpose), cross entropy (cross_entropy), etc.

        Also supports various in-place operations, which don't allow for
        backpropagation, including item assignment (__setitem__), zero assignment (zero_), in place addition (__iadd__), etc.

    Examples:
        Creating and manipulating a tensor:
            >>> x = Tensor([[1, 2], [3, 4]], requires_grad=True)
            >>> y = Tensor([[5, 6], [7, 8]], requires_grad=True)
            >>> z = x + y  # Tensor addition
            >>> z.backward()  # Compute gradients

        Accessing data and gradients:
            >>> print(z.data)  # Access computed data
            >>> print(x.grad)  # Access gradients after backpropagation

    """

    # ----------------------- Initialization -----------------------

    def __init__(self, data, _children=(), requires_grad=False):
        self.data = np.asarray(data)
        self.requires_grad = requires_grad
        self.shape = self.data.shape
        self.ndim = len(self.shape)
        self._prev = set(_children)
        if requires_grad:
            self.grad = zeros_like(self)
            self._backward = lambda: None
        else:
            self.grad = None

    def __repr__(self):
        if self.requires_grad:
            return f'tensor({self.data}, grad={self.grad}, requires_grad={self.requires_grad})'
        # elif self.requires_grad:
        else:
            return f'tensor({self.data})'

    # ----------------------- Basic methods -----------------------
        
    def requires_grad_(self, mode: bool = True):
        self.requires_grad = mode

    def numel(self):
        return self.data.size
    
    def __len__(self):
        return len(self.data)
    
    def __eq__(self, other):
        return Tensor(self.data == other.data)
    
    # We allow Tensor instances to be hashable despite being mutable.
    # This allows using Tensor instances as keys in optimizer state dicts.
    def __hash__(self):
        return id(self)

    # ----------------------- Backward -----------------------

    def backward(self, gradient=None):
        if not self.requires_grad:
            raise RuntimeError('Cannot call backward() on tensor with require_grad=False.')
        
        if gradient is None:
            if self.numel() == 1:
                gradient = ones_like(self)
            else:
                raise RuntimeError('gradient must be specified when calling backward() on tensor with more than one element.')

        # ensure correct shape for gradient
        if gradient.shape != self.shape:
            raise RuntimeError(f'Mismatch in shape: grad_output has a shape of {gradient.shape} and output has a shape of {self.shape}.')
        
        # topological order of the children in the graph
        topo = []
        visited = set()
        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor._prev:
                    if child.requires_grad:
                        build_topo(child)
                topo.append(tensor)
        build_topo(self)

        # go one tensor at a time and apply the chain rule to get its gradient
        self.grad = gradient
        for tensor in reversed(topo):
            tensor._backward()

    # -------------------- Operations requiring backward ----------------------

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad

        out = Tensor(self.data + other.data, (self, other), requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(out.grad.data, self.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(out.grad.data, other.shape)

        if requires_grad:
            out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad

        out = Tensor(self.data * other.data, (self, other), requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(
                    other.data * out.grad.data, 
                    self.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(
                    self.data * out.grad.data,
                    other.shape)

        if requires_grad:
            out._backward = _backward

        return out

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad

        out = Tensor(self.data**other.data, (self, other), requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(
                    other.data * self.data**(other.data - 1) * out.grad.data,
                    self.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(
                    out.data * np.log(self.data) * out.grad.data, 
                    other.shape)

        if requires_grad:
            out._backward = _backward

        return out
    
    def __matmul__(self, other):  # c = a @ b
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad

        out = Tensor(self.data @ other.data, (self, other), requires_grad)

        def transpose(arr):
            """Transpose last two dimensions of array."""
            ndims = arr.ndim
            axes = tuple(range(ndims - 2)) + (ndims - 1, ndims -2)
            return arr.transpose(axes)

        def _backward():
            a = self.data
            b = other.data
            grad_c = out.grad.data 

            # grad_a
            if self.requires_grad:
                if a.ndim == 1 and b.ndim == 1:
                    grad_a = grad_c * b
                    self.grad += Tensor(grad_a)
                elif a.ndim == 1:
                    grad_a = b @ grad_c[..., np.newaxis]
                    ndim = grad_a.ndim
                    axes_to_sum = tuple(range(ndim - 2)) + (ndim - 1,)
                    self.grad += Tensor(grad_a.sum(axes_to_sum))
                elif b.ndim == 1:
                    grad_a = grad_c[..., np.newaxis] @ b[np.newaxis, ...]
                    self.grad += Tensor(grad_a)
                else:
                    grad_a = grad_c @ transpose(b)
                    self.grad += Tensor._unbroadcast(grad_a, self.shape)

            # grad_b
            if other.requires_grad:
                if a.ndim == 1 and b.ndim == 1:
                    grad_b = grad_c * a
                    other.grad += Tensor(grad_b)
                elif a.ndim == 1:
                    grad_b = a[..., np.newaxis] @ grad_c[..., np.newaxis, :]
                    other.grad += Tensor(grad_b)
                elif b.ndim == 1:
                    grad_b = transpose(a) @ grad_c[..., np.newaxis]
                    ndim = grad_b.ndim
                    axes_to_sum = tuple(range(ndim - 2)) + (ndim - 1,)
                    other.grad += Tensor(grad_b.sum(axes_to_sum))
                else:
                    grad_b = transpose(a) @ grad_c
                    other.grad += Tensor._unbroadcast(grad_b, other.shape) 

        if requires_grad:
            out._backward = _backward

        return out

    def transpose(self, dim0, dim1):
        requires_grad = self.requires_grad

        axes = list(range(self.ndim))
        axes[dim0], axes[dim1] = dim1, dim0
        axes = tuple(axes)

        out = Tensor(self.data.transpose(axes), (self,), requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.transpose(dim0, dim1)

        if requires_grad:
            out._backward = _backward

        return out

    def relu(self):
        requires_grad = self.requires_grad

        out = Tensor(np.where(self.data > 0, self.data, 0), 
                     (self,), requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += Tensor(np.where(self.data > 0, out.grad.data, 0))

        if requires_grad:
            out._backward = _backward

        return out 

    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other

    def __rpow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other ** self
    
    def __neg__(self):
        return self * (-1)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def __rmatmul__(self, other):
        return self @ other
    
    def __getitem__(self, key):
        if (isinstance(key, (int, slice, type(Ellipsis))) 
            or (isinstance(key, tuple) and
                all(isinstance(i, (int, slice, type(Ellipsis), type(None))) 
                    for i in key))
            or (isinstance(key, np.ndarray) and 
                key.shape == self.shape and
                all(isinstance(i, bool) for i in key.flatten()))):
            pass
        elif (isinstance(key, tuple) and 
                all(isinstance(t, Tensor) and t.ndim == 1 for t in key)):
            key = tuple([t.data for t in key])
        else:
            raise TypeError('Invalid indexing key for Tensor object.')
        
        requires_grad = self.requires_grad

        out = Tensor(self.data[key], (self,), requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad[key] += out.grad 
        
        if requires_grad:
            out._backward = _backward

        return out

    def flatten(self, start_dim=0, end_dim=-1):

        _children = (self,)
        requires_grad = self.requires_grad 

        if end_dim == -1:
            end_dim = self.ndim - 1

        old_shape = list(self.shape)
        new_shape = old_shape[:]
        new_dim_size = 1
        for dim_size in old_shape[start_dim:end_dim+1]:
            new_dim_size *= dim_size
        new_shape[start_dim:end_dim+1] = [new_dim_size]

        out = Tensor(np.reshape(self.data, new_shape), _children, requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad.data += np.reshape(out.grad.data, old_shape)

        if requires_grad:
            out._backward = _backward

        return out

    @staticmethod
    def stack(tensors: tuple | list, dim=0):

        if isinstance(tensors[0], int):
            tensors = [Tensor(i) for i in tensors]

        _children = tuple(tensors)
        requires_grad = any(t.requires_grad for t in tensors)

        out = Tensor(np.stack([t.data for t in tensors], axis=dim), 
                     _children, requires_grad)
        
        def _backward():
            for n, tensor in enumerate(tensors):
                if tensor.requires_grad:
                    idx = tuple(n if i == dim else slice(None) 
                                for i in range(tensor.ndim))
                    tensor.grad.data += out.grad.data[idx].squeeze()

        if requires_grad:
            out._backward = _backward

        return out

    def max(self, dim):

        _children = (self,)
        requires_grad = self.requires_grad

        out = Tensor(np.max(self.data, dim), _children, requires_grad)

        def _backward():
            if requires_grad:
                
                max_indices = np.argmax(self.data, axis=dim)

                # 1-hot encode index values along new dim-th dimension
                one_hot = np.zeros_like(self.data)
                idxs = [*np.indices(max_indices.shape)]
                idxs.insert(dim, max_indices)
                one_hot[*idxs] = 1

                grad = np.where(one_hot, np.expand_dims(out.grad.data, dim), 0)

                self.grad.data += grad

        if requires_grad:
            out._backward = _backward

        return out

    def exp(self):

        _children = (self,)
        requires_grad = self.requires_grad

        out = Tensor(np.exp(self.data), _children, requires_grad)

        def _backward():
            if requires_grad:

                self.grad += out.grad * out

        if requires_grad:
            out._backward = _backward

        return out
    
    def log(self):

        _children = (self,)
        requires_grad = self.requires_grad

        out = Tensor(np.log(self.data), _children, requires_grad)

        def _backward():
            if requires_grad:

                self.grad += out.grad / self

        if requires_grad:
            out._backward = _backward

        return out

    def sum(self, dim):

        _children = (self,)
        requires_grad = self.requires_grad

        out = Tensor(np.sum(self.data, dim), _children, requires_grad)

        def _backward():
            if requires_grad:

                reps = [1] * self.ndim
                if isinstance(dim, tuple):
                    for d in dim:
                        reps[d] = self.shape[d]
                else:
                    reps[dim] = self.shape[dim]

                self.grad.data += np.tile(np.expand_dims(out.grad.data, dim), 
                                           reps) 

        if requires_grad:
            out._backward = _backward

        return out
    
    def mean(self, dim=None):

        if dim is None:
            dim = tuple(range(self.ndim))

        if isinstance(dim, tuple):
            num_terms = np.prod([self.shape[d] for d in dim], dtype=float)
        elif isinstance(dim, int):
            num_terms = self.shape[dim]

        return Tensor.sum(self, dim) / num_terms
    
    @staticmethod
    def cross_entropy(input, target):
        """Cross entropy between input (logits) and target."""
        
        _children = (input,)
        requires_grad = input.requires_grad

        batch_size = len(input)

        z = input.data
        t = target.data

        z_t = z[np.arange(z.shape[0]), t]
        z_max = np.max(z, axis=1)

        term1 = - (z_t - z_max)
        sum_of_exps = np.sum(np.exp(z - z_max[:, None]), axis=1)
        term2 = np.log(sum_of_exps)

        ce = np.mean(term1 + term2)

        out = Tensor(ce, _children, requires_grad)

        def _backward():
            if requires_grad:
                delta = np.zeros_like(z)
                delta[np.arange(len(t)), t] = 1

                g = np.exp(z - z_max[:, None]) / sum_of_exps[:, None] - delta

                input.grad.data += out.grad.data * g / batch_size

        if requires_grad:
            out._backward = _backward

        return out

    # --------------- In-place operations (no backward needed) -----------------

    __in_place_error = ('A view of a Tensor that requires grad ' 
                        'is being used in an in-place operation.')

    def __setitem__(self, key, value):
        value = value if isinstance(value, Tensor) else Tensor(value)

        if self.requires_grad:
            raise RuntimeError(Tensor.__in_place_error)
        
        self.data[key] = value.data

    def zero_(self):
        if self.requires_grad:
            raise RuntimeError(Tensor.__in_place_error)
        
        self.data[:] = 0

        return self

    def __iadd__(self, other):
        if self.requires_grad:
            raise RuntimeError(Tensor.__in_place_error)
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        self.data += other.data

        return self
    
    def __isub__(self, other):
        if self.requires_grad:
            raise RuntimeError(Tensor.__in_place_error)
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        self.data -= other.data

        return self 

    def __imul__(self, other):
        if self.requires_grad:
            raise RuntimeError(Tensor.__in_place_error)
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        self.data *= other.data

        return self 

    def __itruediv__(self, other):
        if self.requires_grad:
            raise RuntimeError(Tensor.__in_place_error)
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        self.data /= other.data

        return self 

    def __ipow__(self, other):
        if self.requires_grad:
            raise RuntimeError(Tensor.__in_place_error)
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        self.data **= other.data

        return self
    
    def __imatmul__(self, other):
        if self.requires_grad:
            raise RuntimeError(Tensor.__in_place_error)
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        self.data @= other.data

        return self

    # ---------- Non-differentiable operations (no backward needed) ------------

    def argmax(self, dim=None, keepdim=False):
        return Tensor(np.argmax(self.data, dim, keepdims=keepdim))
    
    # ----------------------- Utility Functions -----------------------

    @staticmethod
    def _unbroadcast(input, target_shape):
        """Unbroadcasts input array or Tensor into target shape."""

        array = input if isinstance(input, np.ndarray) else input.data
        array_shape = list(array.shape)
        target_shape = list(target_shape)

        error_message = 'Input is not unbroadcastable into target shape.'

        len_diff = len(array_shape) - len(target_shape)

        # raise error if array_shape is too small relative to target_shaep
        if len_diff < 0:
            raise ValueError(error_message)

        # extend target_shape to match length of array_shape
        extended_shape = [1] * len_diff + target_shape if len_diff > 0 else target_shape

        # add broadcasted dimensions to dims_to_sum
        dims_to_sum = []
        for dim, (n_input, n_shape) in enumerate(zip(array_shape, extended_shape)):
            if n_shape != n_input and n_shape != 1:
                raise ValueError(error_message)
            elif n_shape != n_input and n_shape == 1:
                dims_to_sum.append(dim)

        # trace over dims_to_sum
        array = np.sum(array, axis=tuple(dims_to_sum), keepdims=True)

        # remove first len_diff dimensions
        array = np.squeeze(array, axis=tuple(range(len_diff)))

        return Tensor(array)
    
    
def zeros(*shape, requires_grad=False):
    return Tensor(np.zeros(shape), requires_grad=requires_grad)

def ones(*shape, requires_grad=False):
    return Tensor(np.ones(shape), requires_grad=requires_grad)

def rand(*shape, requires_grad=False):
    return Tensor(np.random.rand(*shape), requires_grad=requires_grad)

def zeros_like(input: (Tensor | np.ndarray), requires_grad=False):
    return Tensor(np.zeros(input.shape), requires_grad=requires_grad)

def ones_like(input: (Tensor | np.ndarray), requires_grad=False):
    return Tensor(np.ones(input.shape), requires_grad=requires_grad)

def rand_like(input: (Tensor | np.ndarray), requires_grad=False):
    return Tensor(np.random.rand(*input.shape), requires_grad=requires_grad)

def arange(*args, requires_grad=False):
    start = 0
    step = 1

    if len(args) == 1:
        end = args[0]
    elif len(args) == 2:
        start, end = args
    elif len(args) == 3:
        start, end, step = args
    else:
        raise TypeError(f'Expected 1 to 3 arguments but got {len(args)}.')

    return Tensor(np.arange(start, end, step), requires_grad=requires_grad)