import numpy as np

matrix = np.array ([[4,17,13], [24,75,66]])

row_mean = np.mean(matrix, axis=1)
print(row_mean)

col_mean = np.mean(matrix, axis=0)
row_sum = np.sum(matrix, axis=1)
col_sum = np.sum(matrix, axis=0)
row_var = np.var(matrix, axis=1)
col_var = np.var(matrix, axis=0)
row_std = np.std(matrix, axis=1)
col_std = np.std(matrix, axis=0)

print(col_mean, row_sum, col_sum, row_var, col_var, row_std, col_std)
