import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv('iris_training.csv', names=column_names)
iris = iris[1:]

x = iris[column_names[:4]].values
y = iris[column_names[4]].values

x = tf.constant(x, dtype=tf.float32)
y = tf.constant(y, dtype=tf.int32)

train_x = x[5:]
train_y = y[5:]

test_x = x[:5]
test_y = y[:5]

model = keras.Sequential([
    keras.layers.Dense(5, input_shape=(4,), activation = 'relu'),
    keras.layers.Dense(5, activation = 'tanh'),
    keras.layers.Dense(3, activation = 'softmax')
])
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy' )


model.fit(train_x, train_y, epochs=100, verbose=1)

predictions = model.predict(test_x)
predicted_classes = np.argmax(predictions, axis=1)

print(predictions)
prediction_score = sum([max(sublist) for sublist in predictions])
accuracy = accuracy_score(test_y, predicted_classes)
print(f'Prediction score: {prediction_score}, Accuracy: {accuracy}')
