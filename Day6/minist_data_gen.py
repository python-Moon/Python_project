from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


##데이터 증강

mnist = keras.datasets.mnist
data = mnist.load_data()

train_data, test_data = data
x_train, y_train = train_data
x_test, y_test = test_data

x_train, x_test = x_train / 255.0, x_test / 255.0

print(len(x_train))

datagen = ImageDataGenerator(
	rotation_range=10,
	width_shift_range=0.1,
	height_shift_range=0.1,
	zoom_range=0.1,
)

train_generator = datagen.flow(x_train.reshape(-1, 28, 28, 1), y_train, batch_size=32)

print(len(train_generator))
test_generator = datagen.flow(x_test.reshape(-1, 28, 28, 1), y_test, batch_size=32)

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

# 데이터 증강으로 생성된 데이터와 기존의 x_train을 합침
augmented_x_train = x_train.reshape(-1, 28, 28, 1)
augmented_y_train = y_train

for i in range(len(train_generator)):
    print(i)
    augmented_images, augmented_labels = train_generator[i]
    augmented_x_train = np.concatenate([augmented_x_train, augmented_images], axis=0)
    augmented_y_train = np.concatenate([augmented_y_train, augmented_labels], axis=0)

print(len(augmented_x_train))

# 모델 훈련
model.fit(augmented_x_train, augmented_y_train, batch_size=32, epochs=5)
model.evaluate(test_generator, verbose=2)
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
