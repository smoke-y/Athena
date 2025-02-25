# Athena

Another deep learning library.
We don't construct AST tree tho. Athena compiles to ops directly which can be run on CPU(numpy) or GPU(custom cuda kernels).

```
AllocTmp: (1, 1), 0
Neg: <Tensor (1, 1) @ 36>
Exp: <Tensor (1, 1) @ 36>
AddS: <Tensor (1, 1) @ 36>
Div: <Tensor (1, 1) @ 38>, <Tensor (1, 1) @ 36>
AllocTmp: (1, 1), 1
```