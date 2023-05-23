import numpy as np
from matplotlib import pyplot as plt


"""Tcs = np.load(r"C:\Users\hansggi\OneDrive - NTNU\BdG\Tcs.npy", allow_pickle=True)
mgs = np.linspace(0, 0.3, 20)
plt.plot(mgs, Tcs, label = "straight")
TcsSkew = np.load(r"C:\Users\hansggi\OneDrive - NTNU\BdG\TcsSkewed.npy", allow_pickle=True)
plt.plot(mgs, TcsSkew, label = "skewed")
plt.legend()
plt.show()

# mgs = np.linspace(0, 2, 20)
# mgs, TcsP = np.load("Parallell_mgdata.npy")
# print(mgs)

# mgs, TcsAP = np.load("AParallell_mgdata.npy")
# print(mgs)
# TcTrue = 0.0971954345703125 # 500 NDelta, 20x20, mg = 0.2
# plt.plot(mgs, TcsP, label = "P")
# plt.plot(mgs, TcsAP, label = "AP")
# plt.legend()
# plt.show()


# mgs, TcsTestP = np.load("testdata\P_ND=15,Ny=5.npy")
# mgs, TcsTestP2 = np.load("testdata\P_ND=100,Ny=5.npy")
# mgs, TcsTestP3 = np.load("testdata\P_ND=200,Ny=5.npy", label = "Par")

# plt.plot(mgs, TcsTestP)
# plt.plot(mgs, TcsTestP2)
# plt.plot(mgs, TcsTestP, label = "P, ND=15")

# mgs, TcsTestAP  = np.load("testdata\AP_ND=15,Ny=5.npy")
# mgs, TcsTestAP2 = np.load("testdata\AP_ND=100,Ny=5.npy")
# mgs, TcsTestAP3 = np.load("testdata\AP_ND=200,Ny=5.npy")

# plt.plot(mgs, TcsTestAP, linestyle = "dashed", label = "AP, ND=15" )
# plt.plot(mgs, TcsTestAP2)
# plt.plot(mgs, TcsTestAP, label = "AP")


mgs, TcsTestP  = np.load(r"C:\Users\hansggi\OneDrive - NTNU\BdG\testdata\Max_P_ND=15,Ny=5.npy")
mgs, TcsTestAP  = np.load(r"C:\Users\hansggi\OneDrive - NTNU\BdG\testdata\Max_AP_ND=15,Ny=5.npy")

# mgs, TcsTestAP2 = np.load("testdata\AP_ND=100,Ny=5.npy")
# mgs, TcsTest = np.load("testdata\AP_ND=200,Ny=5.npy")

plt.plot(mgs, TcsTestP, label = "P_max")
plt.plot(mgs, TcsTestAP, label = "AP_max")

# plt.xlabel("Altermagnet strength m")
# plt.ylabel("Tc, averaging over SC")

# plt.legend()
# plt.show()

mgs, TcsTestP  = np.load(r"C:\Users\hansggi\OneDrive - NTNU\BdG\testdata\Max_P_ND=15,Ny=20.npy")
mgs, TcsTestAP  = np.load(r"C:\Users\hansggi\OneDrive - NTNU\BdG\testdata\Max_AP_ND=15,Ny=20.npy")

# mgs, TcsTestAP2 = np.load("testdata\AP_ND=100,Ny=5.npy")
# mgs, TcsTest = np.load("testdata\AP_ND=200,Ny=5.npy")

# plt.plot(mgs, TcsTestP, label = "P_average")
# plt.plot(mgs, TcsTestAP, label = "AP_average")

plt.xlabel("Altermagnet strength m")
plt.ylabel("Tc, averaging over SC")

plt.legend()
plt.show()"""

mgs, NDelta, Ny, result = np.load("NewData\OneMatND=1Ny)=10mgs10=[(0.0, 0.1)].npy")
