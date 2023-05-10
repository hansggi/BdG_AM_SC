import numpy as np

A = np.array([[1,0,3], [0,5,6]])
B =np.ones_like(A)
print(A)

ind = np.nonzero(A)
print(ind)
B[ind] = A[ind]
print(B)