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