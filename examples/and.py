import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

data = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0,0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
]

from athena import *

PROG.driver = NumpyDriver()

inp = Tensor(data=None, shape = (1,2))
tar = Tensor(data=None, shape = (1,1))

w1 = Tensor.rand((2, 5), sshape=True)
w2 = Tensor.rand((5, 1), sshape=True)
b1 = Tensor.rand((1, 5), sshape=True)
b2 = Tensor.rand((1, 1), sshape=True)

x1 = sigmoid(inp @ w1 + b1)
x2 = sigmoid(x1 @ w2 + b2)

loss = mse(tar, x2)
PROG.compile()
PROG.printForward()

optimizer = SGD([w1, w2, b1, b2], lr=0.01)

for epoch in range(20):
    for i in data:
        inp.load(i[:2])
        tar.load(i[2])
        PROG.forward()
        PROG.backward(x2)
        print(loss.numpy(), x2.numpy())
        optimizer.step()
        PROG.passComplete()