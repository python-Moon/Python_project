import numpy as np
import matplotlib.pyplot as plt

data = np.array([[5, 5], [1.0, 2.0], [5, 2], [7, 3], [2.0, 3.0], [3.0, 4.0], [10.0, 20.0]])
log_transformed_data = np.log1p(data)

print("원본 데이터:\n", data)
print("로그 변환된 데이터:\n", log_transformed_data)

plt.scatter(log_transformed_data[:, 0], log_transformed_data[:,1], label='log Transformed Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
