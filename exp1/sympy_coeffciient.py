import sympy as sp
import time
import pickle
#插值系数
alpha = sp.Symbol('alpha')
beta = sp.Symbol('beta')
gamma = sp.Symbol('gamma')

#顶点坐标 xyz声明
x0, y0, z0 = sp.symbols('x0 y0 z0',constant=True)
x1, y1, z1 = sp.symbols('x1 y1 z1',constant=True)
x2, y2, z2 = sp.symbols('x2 y2 z2',constant=True)
x3, y3, z3 = sp.symbols('x3 y3 z3',constant=True)
x4, y4, z4 = sp.symbols('x4 y4 z4',constant=True)
x5, y5, z5 = sp.symbols('x5 y5 z5',constant=True)
x6, y6, z6 = sp.symbols('x6 y6 z6',constant=True)
x7, y7, z7 = sp.symbols('x7 y7 z7',constant=True)

#顶点坐标组装
v000 = sp.Matrix([x0, y0, z0],constant=True)
v100 = sp.Matrix([x1, y1, z1],constant=True)
v010 = sp.Matrix([x2, y2, z2],constant=True)
v110 = sp.Matrix([x3, y3, z3],constant=True)
v001 = sp.Matrix([x4, y4, z4],constant=True)
v101 = sp.Matrix([x5, y5, z5],constant=True)
v011 = sp.Matrix([x6, y6, z6],constant=True)
v111 = sp.Matrix([x7, y7, z7],constant=True)

#插值函数
fInterp = (1-alpha)*(1-beta)*(1-gamma)*v000 + alpha*(1-beta)*(1-gamma)*v100 + (1-alpha)*beta*(1-gamma)*v010 + alpha*beta*(1-gamma)*v110 + (1-alpha)*(1-beta)*gamma*v001 + alpha*(1-beta)*gamma*v101 + (1-alpha)*beta*gamma*v011 + alpha*beta*gamma*v111

#插值函数求导
fInterpDiffAlpha = sp.diff(fInterp, alpha)
fInterpDiffBeta = sp.diff(fInterp, beta)    
fInterpDiffGamma = sp.diff(fInterp, gamma)

#deformation gradient 拼
F = sp.Matrix([[fInterpDiffAlpha[0], fInterpDiffBeta[0], fInterpDiffGamma[0]],
               [fInterpDiffAlpha[1], fInterpDiffBeta[1], fInterpDiffGamma[1]],
               [fInterpDiffAlpha[2], fInterpDiffBeta[2], fInterpDiffGamma[2]]])


#Green strain
G = 0.5*(F.transpose()*F - sp.eye(3))

#energy density
la = sp.Symbol('la',constant=True)
mu = sp.Symbol('mu',constant=True)

W = la/2*sp.trace(G)**2 + mu*sp.trace(G*G)

#integration
start_time =time.time()
print("start")
W_integrate = sp.integrate(W, (alpha, 0, 1), (beta, 0, 1), (gamma, 0, 1))

end_time = time.time()
execution_time = end_time - start_time
print("积分时间：", execution_time, "秒")

# 保存表达式到文件
with open('exp1/expression.pkl', 'wb') as file:
    pickle.dump(W_integrate, file)
#print(W_integrate)
