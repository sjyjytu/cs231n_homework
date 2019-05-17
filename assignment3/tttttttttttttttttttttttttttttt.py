import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    return x*y

num_iter = 6
lr = 1
x = 1.0
y = 1.0
fs = [1.0,]
xs = [x,]
ys = [y,]
for i in range(num_iter):
    y += lr * x
    x -= lr * y
    xs.append(x)
    ys.append(y)
    fs.append(f(x, y))

iter = [i+1 for i in range(num_iter+1)]
plt.plot(iter, xs,'g--', label='x')
plt.plot(iter, ys,'r--', label='y')
plt.plot(iter, fs,'b--', label='f')
plt.legend()
print(xs)
print(ys)
plt.show()
