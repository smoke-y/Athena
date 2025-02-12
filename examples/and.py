import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from athena import *

inp = Tensor.rand((1,2))
w1 = Tensor.rand((2, 5), sshape=True)
w2 = Tensor.rand((5, 1), sshape=True)
b1 = Tensor.rand((1, 5), sshape=True)
b2 = Tensor.rand((1, 1), sshape=True)
x1 = sigmoid(inp @ w1 + b1)
x2 = sigmoid(x1 @ w2 + b2)
PROG.compile()
PROG.forward()
print(x2.numpy())