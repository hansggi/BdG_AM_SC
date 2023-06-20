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
mu = -0.5
m = 0.0
Delta = 0.5
# sigma = +1
h = 0.
Zup = f(X, Y, mu = mu, m = m, Delta = Delta, sigma = +1, h = h)
Zdn = f(X, Y, mu = mu, m = m, Delta = Delta, sigma = -1, h = h)

minZup = np.min(Zup)
minZdn = np.max(Zdn)
print(minZdn)

fig, ax = plt.subplots(1,2)
# levels = [0.0, -0.2, -0.5, -0.9, -1.5, -2.5, -3.5]
# levels = [-1, 0]
CS = ax[0].contour(X, Y, Zup, levels = [minZup + 0.2])
CS2 = ax[1].contour(X, Y, Zdn, levels = [minZdn - 0.2 ])

ax[0].clabel(CS, inline=1, fontsize=7)
ax[1].clabel(CS2, inline=1, fontsize=7)

# plt.contour(X, Y, Zup, colors='black', levels = 5)#, label ="Up")
# plt.contour(X, Y, Zdn, colors='blue',  levels = 1)#, label ="Dn")
ax[0].set_aspect("equal")
ax[1].set_aspect("equal")

# ax[1].gca().set_aspect('equal')
# plt.title(f"mu={mu} m={m}Delta={Delta}")
plt.xlabel("$k_x$")
plt.ylabel("$k_y$")
# plt.legend()
# plt.savefig("C:/Users/hansggi/OneDrive - NTNU/BdG/figsAMSC/FS3.pdf")
plt.show()