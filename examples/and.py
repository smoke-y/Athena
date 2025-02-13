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

inp = Tensor(data=None, shape = (1,2))
tar = Tensor(data=None, shape = (1,1))

w1 = Tensor(data=None, shape=(2, 5), num=5.0, sshape=True)
w2 = Tensor(data=None, shape=(5, 1), num=5.0, sshape=True)
b1 = Tensor(data=None, shape=(1, 5), num=5.0, sshape=True)
b2 = Tensor(data=None, shape=(1, 1), num=5.0, sshape=True)

x1 = sigmoid(inp @ w1 + b1)
x2 = sigmoid(x1 @ w2 + b2)

loss = mse(tar, x2)
PROG.compile()

for epoch in range(20):
    for i in data:
        inp.load(i[:2])
        tar.load(i[2])
        PROG.forward()
        PROG.backward(x2)