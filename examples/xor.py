import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

data = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
]

from athena import *

PROG.driver = CudaDriver()

inp = Tensor(data=None, shape = (1,2), sshape=True)
tar = Tensor(data=None, shape = (1,1), sshape=True)

w1 = Tensor.rand((2, 5), sshape=True)
w2 = Tensor.rand((5, 1), sshape=True)
b1 = Tensor.rand((1, 5), sshape=True)
b2 = Tensor.rand((1, 1), sshape=True)

x1 = sigmoid(inp @ w1 + b1)
x2 = sigmoid(x1 @ w2 + b2)

loss = mse(tar, x2)
PROG.compile()

optimizer = SGD([w1, w2, b1, b2], lr=0.1)

for epoch in range(3000):
    for i in data:
        inp.load(i[:2])
        tar.load(i[2])
        PROG.forward()
        optimizer.zeroGrad()
        PROG.backward(loss)
        optimizer.step()
        PROG.passComplete()
for i in data:
    inp.load(i[:2])
    PROG.forward()
    print(inp.numpy(), "->", x2.numpy())
    PROG.passComplete()