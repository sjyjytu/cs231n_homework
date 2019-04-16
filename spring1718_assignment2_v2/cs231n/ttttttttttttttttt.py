import numpy as np
# a = np.arange(120).reshape(2,3,4,5)
# c = np.reshape(a,(3,-1))
# print(a.shape)
# # print(b)
a = np.array([1,2])
print(a.shape)
# print(c.shape)
d = a[None,:,None]
print(d)
