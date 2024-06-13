from sklearn. preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

data = np.array([[-5, 5], [1.0, 2.0], [5, 2], [7, 3], [2.0, 3.0], [3.0, 4.0], [10.0, 20.0]])

nomalizer = MinMaxScaler()
nomalized_data = nomalizer. fit_transform(data)

print("원본 데이터\n", data)
print("정규화된 데이터:\n", nomalized_data)

plt.scatter(nomalized_data[:, 0], nomalized_data[:, 1], label = 'Nomalized Data')
plt.title(' Nomalized Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
