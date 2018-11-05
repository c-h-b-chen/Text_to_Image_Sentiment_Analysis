# This is a practice file for learning the ins and outs of PyTorch.

import numpy as np
import torch

# a 5x3 matrix, unitialized
x1 = torch.empty(5, 3)
print('x1', x1)

# Build random 5x3 initialized matrix
x2 = torch.rand(5, 3)
print('x2', x2)

# Build a matrix of zeros of type long
x3 = torch.zeros(5, 3, dtype=torch.long)
print('x3', x3)

# Construct a tensor directly from data
x4 = torch.tensor([5.5, 3])
print('x4', x4)

# Initialize tensor with same size of another tensor. Can change type
x5 = torch.randn_like(x4, dtype=torch.float)
print('x5', x5)

# Get the size of the tensor. Returns a tuple that supports typicall tuple
# operations.
print('x5.size()', x5.size())

# Convert a tensor to a numpy array
print('x5.numpy()', x5.numpy())

# Convert a numpy array to a tensor
x6 = np.array([1, 2, 3, 4])
print('x6', x6)
x7 = torch.from_numpy(x6)
print('x7', x7)

# Note: all tensors on cpu support tensor to np.array conversion besides
# CharTensor

# CUDA Tensors
# Tensors can be moved to any device using the .to method.
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x7, device=device)  # directly create a tensor on GPU
    x8 = x7.to(device)                       # or just use strings
    z = x8 + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also
