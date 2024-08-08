import numpy as np
import cv2
from matplotlib import pyplot as plt
image_path= 'chungun.png'
image = cv2.imread(image_path)


filter = np.array([
    [1, 1, 1],
    [1, -10, 1],
    [1, 1, 1]
])

feature_map = cv2.filter2D(image, -1, filter)

plt.subplot(121), plt.imshow(image), plt.title('Original Image')
plt.subplot(122), plt.imshow(feature_map), plt.title('Feature Map')
plt.show()
