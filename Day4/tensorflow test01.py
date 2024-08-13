from tensorflow import keras

import numpy as np

model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(1,), activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer = 'adam', loss='mean_squared_error')

X = np.array([1,2,3,4])
Y = 2 * X + 1

model.fit(X, Y, epochs=1000, verbose=0)
prediction = model.predict([6])
print(f'predction: {prediction[0][0]}')
