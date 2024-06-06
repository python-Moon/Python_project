import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 10000)
y = x**2 -4*x + 5

plt.plot(x, y, label = "line Chart")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()
