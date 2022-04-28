import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1e-6,1e2,1000)
y = 1 - 1/(np.power(x,3) + 1)

plt.plot(x, y)
plt.show()

