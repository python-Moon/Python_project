import numpy as np

matrix = np.array([[1,2], [3,4]])

result_add = matrix + 2
result_square = np.square(matrix)
result_multi = np.dot(matrix, matrix)

print(result_add)
print(result_multi)
