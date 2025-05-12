import numpy as np

class Driver:
    def __init__(self) -> None:
        self._mem = []
        self.sshapeLen = 0
        self.tmp = []
    def compile(self) -> None:
        x = len(self._mem)
        self.sshapeLen = x
        for i in range(len(self.tmp)): self.tmp[i].id = i + x
        del self.tmp
    def allocTmpComp(self, tens) -> None: self.tmp.append(tens)
    def allocTmp(self, num: float, shape: tuple, tens) -> None: raise NotImplementedError(f"allocTmp not implemented for {self.__class__.__name__}")
    def allocObj(self, obj: np.ndarray, tens) -> None: raise NotImplementedError(f"allocObj not implemented for {self.__class__.__name__}")
    def allocNum(self, num: float, shape: tuple, tens) -> None: raise NotImplementedError(f"allocNum not implemented for {self.__class__.__name__}")
    def numpy(self, id: int, out: np.ndarray) -> None: raise NotImplementedError(f"numpy not implemented for {self.__class__.__name__}")
    def fill(self, id: int, out: np.ndarray) -> None: raise NotImplementedError(f"fill not implemented for {self.__class__.__name__}")
    def load(self, id: int, data: np.ndarray) -> None: raise NotImplementedError(f"load not implemented for {self.__class__.__name__}")
    def passComplete(self) -> None: raise NotImplementedError(f"passComplete not implemented for {self.__class__.__name__}")
