from tensorflow import keras
from PIL import Image
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

##다른사진사용(내손글씨><)


mnist = keras.datasets.mnist
data = mnist.load_data()

train_data, test_data = data
x_train, y_train = train_data
x_test, y_test = test_data

my_x = []
my_y = [5, 3, 1, 7, 8]
for i in range(5):
        image = Image.open(f"./number_test0{i+1}.jpg").convert("L")
        image = image.resize((28,28))
        image_array = np.array(image) / 255.0
        image_array = np.reshape(image_array, (1, 28, 28, 1))
        my_x.append(image_array)


my_x = [image[0, :, :, 0]for image in my_x]
my_x = np.array(my_x).reshape(-1, 28, 28, 1)
my_x = 1 - my_x
my_x[my_x <= 0.05] = 0
my_x[my_x > 0.05] += 0.3
my_x[my_x > 1] = 1


x_train = x_train / 255.0

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
predictions = model.predict(my_x)

plt.figure(figsize=(10,10))
num_images = 16
count = 0
print(predictions)

for i in range(len(my_x)):
        plt.subplot(4, 4, count + 1)
        plt.imshow(my_x[i], cmap='gray')
        plt.title(f"Actual:{my_y[i]}, Predicted: {np.argmax(predictions[i])}")
        plt.axis('off')
        count +=1



plt.show()
