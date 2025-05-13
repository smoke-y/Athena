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
print(src.grad.numpy())
