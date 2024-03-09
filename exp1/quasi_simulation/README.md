fas 方法  在粗网格上,fix point应该怎么表达?

振东师兄的论文  将newton-multigrid方法里固定的A,b每次smooth以后都可以用非线性方程更新一下,
A0c0 =r0 is the current linearized system
We apply this correction to x0 after proper scaling, update the residual r0 (and the matrix A0 if needed)