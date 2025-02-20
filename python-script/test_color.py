import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential
import os
from cv2 import imread,resize

def load_data(data_dir, image_size):
    images = []
    for filename in os.listdir(data_dir):
        image = imread(os.path.join(data_dir, filename))
        image = resize(image, image_size)
        images.append(image)
    return np.array(images)


# CNN modeli oluşturulması
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(2, (3, 3), activation='tanh', padding='same'), # Çıkış kanalı 2 (renk kanalları)
    UpSampling2D((2, 2))
])

# Modelin derlenmesi
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Eğitim veri kümesi oluşturulması (örnek olarak, rastgele gri tonlamalı görüntüler kullanılır)
num_samples = 1000
image_size = (128, 128) # Görüntü boyutu

X_train = load_data("F:\\Restoresyon\\data\\buildings_gray", image_size)
y_train = load_data("F:\\Restoresyon\\data\\buildings", image_size)

print("eğitime başlanacak")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

