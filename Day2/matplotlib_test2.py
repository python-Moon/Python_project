import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [10, 15, 7, 25]

plt.plot(x, y, label="test")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
