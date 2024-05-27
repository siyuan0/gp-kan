import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


for mx in [-3]:
    for sx in [1e-3, 1e-2, 0.1, 0.5, 1, 10]:
        mz = np.tanh(mx)
        sz = np.sqrt(sigmoid(sx**2 - mx**2 + 4) / 4)
        z = np.random.normal(mz, sz, size=[10000])

        plt.hist(z, bins=100, label=f"mx={mx}, sx={sx}", alpha=0.3)
plt.legend()
plt.savefig("dist.png")
