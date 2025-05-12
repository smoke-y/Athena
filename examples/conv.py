import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from athena import *

PROG.driver = NumpyDriver()

kernel1d = Tensor([1,1,1])
src1d    = Tensor([1,2,3,4,5,6])

t1d = conv1d(src1d, kernel1d)

src = Tensor([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
])
kernel = Tensor([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

t = conv2d(src, kernel)

PROG.compile()
PROG.forward()

PROG.backward(t)

print(t.numpy())
print(t.grad.numpy())

import torch
import torch.nn.functional as F

# Define the image (5x5) as a tensor
image = torch.tensor([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=torch.float32)

# Define the kernel (3x3) as a tensor
kernel = torch.tensor([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=torch.float32, requires_grad=True)

# PyTorch expects inputs in (batch, channel, height, width) format
# Add batch and channel dimensions (both 1 in this case)
image = image.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 5, 5)
kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 3, 3)
kernel.retain_grad()

# Perform convolution
result = F.conv2d(image, kernel, stride=1, padding=0)

# Remove batch and channel dimensions for comparison
result = result.squeeze()
result.backward(torch.ones_like(result))

print(result)
print(kernel.grad)
