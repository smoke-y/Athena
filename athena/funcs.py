from numpy import shape
from .tensor import *

def sigmoid(x: Tensor) -> Tensor:
    PROG.fa([
        Neg(x, t := Tensor(data=None, shape=x.shape)),
        Exp(t, t),
        AddS(t, 1, t),
        Div(Tensor(data=None, shape=x.shape, num=1), t, t),
    ])
    PROG.ba([
        Sub(Tensor(data=None, shape=x.shape, num=1), t, tmp := Tensor(data=None, shape=x.shape)),
        Mul(t, tmp, tmp),
        Mul(tmp, t.grad, x.grad),
    ])
    return t

def mse(target: Tensor, x: Tensor) -> Tensor: return ((x - target) ** 2) / target.numel()

def conv1d(src: Tensor, kernel: Tensor) -> Tensor: 
    PROG.f(Conv1D(src, kernel, t:=Tensor(data=None, shape=src.shape[0] - kernel.shape[0] + 1)))
    PROG.ba([Conv1DBack(src, kernel, t.grad)])
    return t
def conv2d(src: Tensor, kernel: Tensor) -> Tensor:
    PROG.f(Conv2D(src, kernel, t:=Tensor(data=None, shape=(src.shape[0]-kernel.shape[0]+1, src.shape[1]-kernel.shape[1]+1))))
    PROG.ba([Conv2dBack(src, kernel, t.grad)])
    return t
