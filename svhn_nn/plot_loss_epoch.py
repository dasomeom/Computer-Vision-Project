import matplotlib.pyplot as plt
import numpy as np


arr = np.load("loss_epoc_array.npy")
arr2 = np.load("loss_epoc_array_test.npy")

print(arr[:, 0])
plt.plot(arr[:, 0], arr[:, 1], '-', color='red', label='train')
plt.plot(arr2[:, 0], arr2[:, 1], '-', color='blue', label='test')
plt.title("Loss Function vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.show()

