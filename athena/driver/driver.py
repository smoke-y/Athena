import numpy as np

class Singleton(object):
    _instance = None
    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance
    
class Driver:
    def __init__(self) -> None:
        super().__init__()
        self._mem = []
        self.static = []
        self.temp = []
        self.sshapeLen = 0
    def allocateObj(self, obj: np.ndarray, sshape: bool, tens) -> None:
        self.static.append([tens, obj]) if sshape else self.temp.append([tens, obj])
    def allocateNum(self, num: float, shape: tuple, sshape: bool, tens) -> None:
        self.static.append([tens, num, shape]) if sshape else self.temp.append([tens, num, shape])
    def numpy(self, id: int, out: np.ndarray) -> None: raise NotImplementedError(f"numpy not implemented for {self.__class__.__name__}")
    def fill(self, id: int, out: np.ndarray) -> None: raise NotImplementedError(f"fill not implemented for {self.__class__.__name__}")
    def load(self, id: int, data: np.ndarray) -> None: raise NotImplementedError(f"load not implemented for {self.__class__.__name__}")
    def compile(self) -> None: raise NotImplementedError(f"compile not implemented for {self.__class__.__name__}")
    def passComplete(self) -> None: raise NotImplementedError(f"passComplete not implemented for {self.__class__.__name__}")