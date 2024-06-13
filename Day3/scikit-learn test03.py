from sklearn. preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

data = np.array([[-5, 5], [1.0, 2.0], [5, 2], [7, 3], [2.0, 3.0], [3.0, 4.0], [10.0, 20.0]])

minmax_scaler = MinMaxScaler(feature_range=(1,100))
minmax_scaled_data = minmax_scaler.fit_transform(data)

print("원본 데이터:\n", data)
print("최소-최대 스케일링 데이터:\n", minmax_scaled_data)

plt.scatter(minmax_scaled_data[:, 0], minmax_scaled_data[:,1], label="MinMax Scaled Data")
plt.title("MinMax Scaled Data")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
