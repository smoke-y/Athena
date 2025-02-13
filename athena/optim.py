from .ops import *
from .program import *

class Optimizer:
    def __init__(self, params: list) -> None:
        self.params = params
    def step(self) -> None: raise NotImplementedError("step not implemented")

class SGD(Optimizer):
    def __init__(self, params: list, lr: float) -> None:
        super().__init__(params)
        for param in params:
            PROG.oa([
                MulS(param.grad, lr, param.grad),
                Sub(param, param.grad, param),
                Fill(param.grad, 0.0)
            ])
    def step(self) -> None:
        for instr in PROG.optimizerSet: instr.forward()