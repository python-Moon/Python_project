from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np


##원래꺼(test)

mnist = keras.datasets.mnist
data = mnist.load_data()

train_data, test_data = data
x_train, y_train = train_data
x_test, y_test = test_data

x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential([
    layers.Reshape((28, 28, 1)),
    layers.Conv2D(16, 3, activation = 'relu', padding = 'same', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(128),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)
predictions = model.predict(x_test)

plt.figure(figsize=(10,10))
num_images = 16
count = 0
print(predictions)

for i in range(len(x_test)):
        if y_test[i] !=np.argmax(predictions[i]):
            plt.subplot(4, 4, count+1)
            plt.imshow(x_test[i], cmap='gray')
            plt.title(f"Actual:{y_test[i]}, Predicted: {np.argmax(predictions[i])}")
            plt.axis('off')
            count +=1

            if count == num_images:
                break

plt.show()

