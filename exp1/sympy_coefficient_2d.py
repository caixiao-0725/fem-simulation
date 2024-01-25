import sympy as sp
import time
import pickle

alpha = sp.Symbol('alpha')
beta = sp.Symbol('beta')



x00, y00 = sp.symbols('x00 y00',constant=True)
x10, y10 = sp.symbols('x10 y10',constant=True)
x01, y01 = sp.symbols('x01 y01',constant=True)
x11, y11 = sp.symbols('x11 y11',constant=True)

v00 = sp.Matrix([x00, y00],constant=True)
v10 = sp.Matrix([x10, y10],constant=True)
v01 = sp.Matrix([x01, y01],constant=True)
v11 = sp.Matrix([x11, y11],constant=True)

fInterp = (1-alpha)*(1-beta)*v00 + alpha*(1-beta)*v10 + (1-alpha)*beta*v01 + alpha*beta*v11

fInterpDiffAlpha = sp.diff(fInterp, alpha)

fInterpDiffBeta = sp.diff(fInterp, beta)

F = sp.Matrix([[fInterpDiffAlpha[0], fInterpDiffBeta[0]],
                [fInterpDiffAlpha[1], fInterpDiffBeta[1]]])

#Green strain
G = 0.5*(F.transpose()*F - sp.eye(2))

#energy density
la = sp.Symbol('la',constant=True)
mu = sp.Symbol('mu',constant=True)

start_time =time.time()
W = la/2*sp.trace(G)**2 + mu*sp.trace(G*G)

W_integrate = sp.integrate(W, (alpha, 0, 1), (beta, 0, 1))

end_time = time.time()
execution_time = end_time - start_time
print("积分时间：", execution_time, "秒")


# 保存表达式到文件
with open('exp1/expression.pkl', 'wb') as file:
    pickle.dump(W_integrate, file)
#print(W_integrate)

             