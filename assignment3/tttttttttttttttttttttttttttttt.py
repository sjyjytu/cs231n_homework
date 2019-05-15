import numpy as np

a = np.zeros((3,4))
b = np.ones((3, ))
for i in range(a.shape[1]):
    a[:,i] = b
    b += b
print(a)