import pickle
import sympy as sp

# 创建符号变量和表达式
x = sp.Symbol('x')
expr = x**2 + 2*x + 1

# 保存表达式到文件
with open('expression.pkl', 'wb') as file:
    pickle.dump(expr, file)

# 加载表达式
with open('expression.pkl', 'rb') as file:
    loaded_expr = pickle.load(file)

# 打印加载的表达式\
a = sp.integrate(loaded_expr, (x, 0, 1))
print(a)