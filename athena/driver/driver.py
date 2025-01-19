import numpy as np

class Singleton(object):
    _instance = None
    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance
    
class Driver:
    _mem = []
    def allocateObj(self, obj: np.ndarray) -> int: raise NotImplementedError(f"allocateObj not implemented for {self.__class__.__name__}")
    def allocateNum(self, num: float, shape: tuple) -> int: raise NotImplementedError(f"allocateNum not implemented for {self.__class__.__name__}")
    def allocateTemp(self, num: float, shape: tuple) -> int: raise NotImplementedError(f"allocateTemp not implemented for {self.__class__.__name__}")
    def numpy(self, id: int, out: np.ndarray) -> None: raise NotImplementedError(f"numpy not implemented for {self.__class__.__name__}")
    def fill(self, id: int, out: np.ndarray) -> None: raise NotImplementedError(f"fill not implemented for {self.__class__.__name__}")
    def compile(self) -> None: raise NotImplementedError(f"compile not implemented for {self.__class__.__name__}")