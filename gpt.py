import numpy as np
from matplotlib import pyplot as plt
Nx, Ny = 20, 10
def f(ix, theta):
    return Ny/2 + np.tan(theta)*(ix - Nx/2) 

V = np.arange(8)

# a = V[::4]
# b = V[1::4]

# c = V[2::4]
# d = V[3::4]

# print(a)
# print(b)
# print(c)
# print(d)
print(V)
V2 = np.reshape(V, (2, 4))
print(V2)
print(V2.reshape(((8,))))