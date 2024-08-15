# fem-simulation
finite element method simulation.
腾讯动画仿真方向课题12：AI加速的物理模拟算法

处理的网格问题如下

![image](https://github.com/caixiao-0725/fem-simulation/blob/main/result/mesh.png )

## exp1:

benchmark   FAS and Newton Multigrid.

dynamic的效果

![image](https://github.com/caixiao-0725/fem-simulation/blob/main/result/dynamic.gif )   

quasi-dynamic的收敛结果对比图，可以看到使用FAS后是力的无穷范数是线性下降的

![image](https://github.com/caixiao-0725/fem-simulation/blob/main/result/exp1.png )  

## exp2:

optimization of I

关于优化插值矩阵的推导在result/I.pdf中，对于有固定点的情况，针对固定点附近的I进行优化，提升收敛效果比较明显

## exp3:

use network to solve Ax=b.

network设计如下
![image](https://github.com/caixiao-0725/fem-simulation/blob/main/result/exp3.png )  