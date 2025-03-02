# Athena

Another deep learning library.
We don't construct AST tree tho. Athena compiles to ops directly which can be run on CPU(numpy) or GPU(custom cuda kernels).

```python
from athena import *

PROG.driver = CudaDriver()

'''
Shape Static Tensor
By marking a tensor to be static(sshape=True) in shape, the tensor
is allocated in the beginning on the device and is not freed until the end
'''
x = Tensor(data=None, shape = (1,2), num=3, sshape=True)
v = Tensor(data=[[1,2]])
d = x + v

#compile the above operations
PROG.compile()

#print the ops generated in the forward pass
PROG.printForward()
#print the ops generated in the backward pass
PROG.printBackward()

PROG.forward()
print(d.numpy())  #no need to .detach()
PROG.backward(d)  #backward with respect to Tensor d
```
```
AllocTmp: (1, 2), 0
AllocTmp: (1, 2), 0
AllocTmp: (1, 2), 0
Add: <Tensor (1, 2) @ 0>, <Tensor (1, 2) @ 2>
Add: <Tensor (1, 2) @ 1>, <Tensor (1, 2) @ 5>
Add: <Tensor (1, 2) @ 3>, <Tensor (1, 2) @ 5>
[[4. 5.]]
```