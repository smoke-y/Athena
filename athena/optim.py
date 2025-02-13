class Optimizer:
    def __init__(self, params: list) -> None:
        self.params = params
    def step(self) -> None: raise NotImplementedError("step not implemented")

class SGD(Optimizer):
    def __init__(self, params: list, lr: float) -> None:
        super.__init__(self, params)
        self.lr = lr
    def step(self) -> None: pass