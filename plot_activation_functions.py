import numpy as np
from matplotlib import pyplot as plt
import tikzplotlib

 
# plot a sigmoid function
x = np.linspace(-10, 10, 100)
y = 1 / (1 + np.exp(-x))
plt.rcParams['lines.linewidth'] = 2
plt.plot(x, y)
plt.title("Sigmoid function")
# grid
plt.grid()
# put the formula in the plot
plt.text(-9, 0.9, r'$\sigma(x) = \frac{1}{1 + e^{-x}}$', fontsize=20)
plt.tight_layout()
# plt.savefig("sigmoid.png", dpi=600)
tikzplotlib.save("sigmoid.tex")
plt.show()

# plot a tanh function
x = np.linspace(-10, 10, 100)
y = np.tanh(x)
plt.rcParams['lines.linewidth'] = 2
plt.plot(x, y)
plt.title("Tanh function")
plt.grid()
plt.text(-9, 0.9, r'$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$', fontsize=20)
plt.tight_layout()
# plt.savefig("tanh.png", dpi=600)
tikzplotlib.save("tanh.tex")
plt.show()

# plot a relu function
x = np.linspace(-10, 10, 100)
y = np.maximum(x, 0)
plt.rcParams['lines.linewidth'] = 2
plt.plot(x, y)
plt.title("ReLU function")
plt.grid()
plt.text(-9, 9, r'$ReLU(x) = max(0, x)$', fontsize=20)
plt.tight_layout()
# plt.savefig("relu.png", dpi=600)
tikzplotlib.save("relu.tex")
plt.show()