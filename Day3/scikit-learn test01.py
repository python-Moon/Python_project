from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

data = np. array([[-5, -5], [2.0, 6.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [10.0, 20.0]])

scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)

print("원본 데이터:\n", data)
print("표준화된 데이터\n", standardized_data)

plt.scatter(standardized_data[:, 0], standardized_data[:, 1], label="standardized Data")
plt.title('Standardized Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
