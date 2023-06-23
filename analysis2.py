import numpy as np
from matplotlib import pyplot as plt
def getms(items, el):
    n = len(items)
    ms = np.zeros(n)
    for i in range(n):
        print(items[i])
        ms[i] = items[i][el]
        print(ms[i])
    
    return ms
NDelta = 10
itemsssAMF, TcsssAMF = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/straightskewed/AMFalse(20, 20, {NDelta}, 0, 0, 1.0, False, 0, 0).npy", allow_pickle=True)
mgssAMF = getms(itemsssAMF, 5)
itemsssAMT, TcsssAMT = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/straightskewed/AMTrue(20, 20, {NDelta}, 0, 0, 1.0, True, 0, 0).npy", allow_pickle=True)
mgssAMT = getms(itemsssAMT, 5)

itemsssFMF, TcsssFMF = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/straightskewed/FMFalse(20, 20, {NDelta}, 1.0, 0, 0, False, 0, 0).npy", allow_pickle=True)
mgssFMF = getms(itemsssFMF, 3)

fig, ax = plt.subplots()

ax.plot(mgssAMF, TcsssAMF, label = "AMF")
ax.plot(mgssAMT, TcsssAMT, label = "AMT")
ax.plot(mgssFMF, TcsssFMF, label = "FMF")
plt.legend()

plt.tight_layout()
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
    })

plt.show()