import numpy as np
from tensorflow import keras

x = np.arange(-20, 21)
y = x ** 2 + 3

test_x = np.array([4.5, -3.2, 6.7, 8.9])
test_y = test_x ** 2 + 3

model = keras.Sequential([
    keras.layers.Dense(3, input_shape=(1,), activation='tanh'),
    keras.layers.Dense(3, activation='linear'),
    keras.layers.Dense(3, activation='tanh')
])

model.compile(optimizer = 'adam', loss='mean_squared_error')

model.fit(x, y, epochs=1000, verbose=0)
predictions = model.predict(test_x)
diff_sum = 0

for i in range(len(test_x)):
    tx = test_x[i]
    ty = test_y[i]
    py = predictions[i][0]
    diff_sum += abs(py-ty)
    print(f'Input: {tx}, Predict : {py}, Answer: {ty}, Diff: {abs(py - ty)}')

print(f'오차의 합: {diff_sum}')
