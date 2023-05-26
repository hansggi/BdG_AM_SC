import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

def f(kx, ky, mu, m, Delta, sigma, h):
    # if Delta == 0:
    #     return - mu   - 2 * ( np.cos(kx) + np.cos(ky)) - 2 * m*( np.cos(kx) - np.cos(ky)) 

    # else:
    return - h - 2 * m*( np.cos(kx) - np.cos(ky)) + sigma* np.sqrt( (- mu - 2*( np.cos(kx) + np.cos(ky)))**2 + Delta**2)
x = np.linspace(-np.pi, np.pi, 500)
y = np.linspace(-np.pi, np.pi, 500)

X, Y = np.meshgrid(x, y)
mu = -0.
m = 0.06
Delta = 0.2
# sigma = +1
h = 0.
Zup = f(X, Y, mu = mu, m = m, Delta = Delta, sigma = +1, h = h)
Zdn = f(X, Y, mu = mu, m = m, Delta = Delta, sigma = -1, h = h)

fig, ax = plt.subplots()
levels = [0.0, -0.2, -0.5, -0.9, -1.5, -2.5, -3.5]
levels = [-1, 0]
CS = ax.contour(X, Y, Zdn, levels = 3)
ax.clabel(CS, inline=1, fontsize=10)

# plt.contour(X, Y, Zup, colors='black', levels = 5)#, label ="Up")
# plt.contour(X, Y, Zdn, colors='blue',  levels = 1)#, label ="Dn")

plt.gca().set_aspect('equal')
# plt.title(f"mu={mu} m={m}Delta={Delta}")
# plt.xlabel("$k_x$")
# plt.ylabel("$k_y$")
# plt.legend()
# plt.savefig("C:/Users/hansggi/OneDrive - NTNU/BdG/figsAMSC/FS3.pdf")
plt.show()