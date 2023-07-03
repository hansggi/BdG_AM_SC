import numpy as np
from matplotlib import pyplot as plt
from icecream import ic
def getms(items, el):
    n = len(items)
    ms = np.zeros(n)
    for i in range(n):
        # print(items[i])
        ms[i] = items[i][el]
        # print(ms[i])
    
    return ms

# NDelta = 10
# itemsssAMF, TcsssAMF = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/straightskewed/AMFalse(20, 20, {NDelta}, 0, 0, 1.0, False, 0, 0).npy", allow_pickle=True)
# mgssAMF = getms(itemsssAMF, 5)
# itemsssAMT, TcsssAMT = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/straightskewed/AMTrue(20, 20, {NDelta}, 0, 0, 1.0, True, 0, 0).npy", allow_pickle=True)
# mgssAMT = getms(itemsssAMT, 5)

# itemsssFMF, TcsssFMF = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/straightskewed/FMFalse(20, 20, {NDelta}, 1.0, 0, 0, False, 0, 0).npy", allow_pickle=True)
# mgssFMF = getms(itemsssFMF, 3)

# fig, ax = plt.subplots()

# ax.plot(mgssAMF, TcsssAMF, label = "AMF")
# ax.plot(mgssAMT, TcsssAMT, label = "AMT")
# ax.plot(mgssFMF, TcsssFMF, label = "FMF")
# plt.legend()

# plt.tight_layout()
# plt.rcParams.update({
#         "text.usetex": True,
#         "font.family": "Times",
#     })

# plt.show()


NDelta = 10
itemsssAM, TcsssImps_AM = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/imps/AM0.00((20, 20), 10, 1.0, 0.2).npy", allow_pickle=True)
mgsImps = getms(itemsssAM, 5)
TcsAv = np.average(TcsssImps_AM)
print(TcsAv)
# dTcs= np.sqrt(np.mean(TcsssImps_AM**2) - np.mean(TcsssImps_AM)**2)
dTcs = np.std(TcsssImps_AM)
print(dTcs)
# print(itemsssAM)
# print(mgsImps)
Tc0AM = 0.0952
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        'font.size'   : 28
    })

# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off
fig0, ax0 = plt.subplots(1,3, sharey=True, figsize = (15, 5))

ax0[0].set_xticklabels([])
ax0[1].set_xticklabels([])
ax0[2].set_xticklabels([])
ax0[0].set_ylabel("$T_c$", fontsize = 32)
fig, ax = plt.subplots()

ax.plot(np.arange(len(mgsImps)), TcsssImps_AM, label = "$T_{c,i}$")
ax.axhline(y=np.average(TcsssImps_AM), label = r"$\langle T_c \rangle_{i}$", color = "orange")
ax.axhline(y=Tc0AM, label = "$T_{c,0}$", color = "green")
ax.axhline(y = TcsAv + dTcs, color = "orange", ls = "dashed", lw = 1.)
ax.axhline(y = TcsAv - dTcs, color = "orange", ls = "dashed", lw = 1.)

ax0[0].plot(np.arange(len(mgsImps)), TcsssImps_AM, label = "$T_{c,i}$")
ax0[0].axhline(y=np.average(TcsssImps_AM), label = r"$\langle T_c \rangle_{i}$", color = "orange")
ax0[0].axhline(y=Tc0AM, label = "$T_{c,0}$", color = "green")
ax0[0].axhline(y = TcsAv + dTcs, color = "orange", ls = "dashed", lw = 1.)
ax0[0].axhline(y = TcsAv - dTcs, color = "orange", ls = "dashed", lw = 1.)


ax.set_ylabel("$T_c$")
ax.set_xlabel("$i$")
plt.legend(loc = "best", fontsize = "small")

plt.tight_layout()
# plt.rcParams.update({
#         "text.usetex": True,
#         "font.family": "Times",
#     })

# plt.show()

NDelta = 10
itemsssAM, TcsssImps_AM, Tc0AM = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/imps/AMm=0.5mz=0((20, 20), 10, 1.0, 0.2).npy", allow_pickle=True)
mgsImps = getms(itemsssAM, 5)
# print(mgsImps)
TcsAv = np.average(TcsssImps_AM)
dTcs = np.std(TcsssImps_AM)
ic(dTcs)
ic(Tc0AM, TcsAv, TcsAv / Tc0AM)
# plt.rcParams.update({
#         "text.usetex": True,
#         "font.family": "Times",
#         'font.size'   : 28
#     })
fig, ax = plt.subplots()


ax.plot(np.arange(len(mgsImps)), TcsssImps_AM, label = "$T_{c,i}$")
ax.axhline(y=np.average(TcsssImps_AM), label = r"$\langle T_c \rangle_{i}$", color = "orange")
ax.axhline(y=Tc0AM, label = "$T_{c,0}$", color = "green")
ax.axhline(y = TcsAv + dTcs, color = "orange", ls = "dashed", lw = 1.)
ax.axhline(y = TcsAv - dTcs, color = "orange", ls = "dashed", lw = 1.)

ax0[1].plot(np.arange(len(mgsImps)), TcsssImps_AM, label = "$T_{c,i}$")
ax0[1].axhline(y=np.average(TcsssImps_AM), label = r"$\langle T_c \rangle_{i}$", color = "orange")
ax0[1].axhline(y=Tc0AM, label = "$T_{c,0}$", color = "green")
ax0[1].axhline(y = TcsAv + dTcs, color = "orange", ls = "dashed", lw = 1.)
ax0[1].axhline(y = TcsAv - dTcs, color = "orange", ls = "dashed", lw = 1.)

ax.set_ylabel("$T_c$")
ax.set_xlabel("$i$")
plt.legend(loc = "best", fontsize = "small")

plt.tight_layout()
# plt.rcParams.update({
#         "text.usetex": True,
#         "font.family": "Times",
#         'font.size'   : 22
#     })

# plt.show()

NDelta = 10
itemsssFM, TcsssImps_FM, Tc0FM = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/imps/FMm=0.0mz=0.5((20, 20), 10, 1.0, 0.2).npy", allow_pickle=True)
mzsImps = getms(itemsssFM, 3)
TcsAv = np.average(TcsssImps_FM)
dTcs = np.std(TcsssImps_FM)
ic(dTcs)
ic(Tc0FM, TcsAv, TcsAv / Tc0FM)
# print(mzsImps)
fig, ax = plt.subplots()

ax.plot(np.arange(len(mzsImps)), TcsssImps_FM, label = "$T_{c,i}$")
ax.axhline(y=np.average(TcsssImps_FM), label = r"$\langle T_c \rangle_{i}$", color = "orange")
ax.axhline(y=Tc0FM, label = "$T_{c,0}$", color = "green")
ax.axhline(y = TcsAv + dTcs, color = "orange", ls = "dashed", lw = 1.)
ax.axhline(y = TcsAv - dTcs, color = "orange", ls = "dashed", lw = 1.)

ax0[2].plot(np.arange(len(mzsImps)), TcsssImps_FM, label = "$T_{c,i}$")
ax0[2].axhline(y=np.average(TcsssImps_FM), label = r"$\langle T_c \rangle_{i}$", color = "orange")
ax0[2].axhline(y=Tc0FM, label = "$T_{c,0}$", color = "green")
ax0[2].axhline(y = TcsAv + dTcs, color = "orange", ls = "dashed", lw = 1.)
ax0[2].axhline(y = TcsAv - dTcs, color = "orange", ls = "dashed", lw = 1.)


ax.set_ylabel("$T_c$")
ax.set_xlabel("$i$")
plt.legend(loc = "best", fontsize = "small")

plt.tight_layout()
# plt.rcParams.update({
#         "text.usetex": True,
#         "font.family": "Times",
#     })

plt.show()


