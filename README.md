# MyTorch

A simplified version of PyTorch for constructing and training neural networks. This project was inspired and builds upon Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) package to allow for multidimensional Tensors. It also providing functionality for neural network construction, data handling, and optimization.

The components of MyTorch are:
- engine.py: defines the Tensor class, the foundation of MyTorch
- nn.py: implementation of Module base class for MyTorch layers, and a few useful subclasses
- data.py: tools for processing data for use with neural networks
- optim.py: tools for neural network optimization

The notebook experiments.ipynb contains a demonstration of MyTorch in action: we train a neural network to classify MNIST images and compare its performance to an analogous network developed in PyTorch.

## Example Usage

Constructing and training an MNIST image classifier:

```python
from mytorch import nn, data, optim

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

dataset = data.MNIST(train=True)
dataloader = data.DataLoader(dataset, batch_size=32)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

for images, labels in dataloader:
    optimizer.zero_grad()            # set gradients to zero
    output = net(images)             # output of NN
    loss = loss_fn(output, labels)   # compute loss
    loss.backward()                  # backpropagate gradients
    optimizer.step()                 # optimizer step
```
