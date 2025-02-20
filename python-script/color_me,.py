# %%
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from skimage.color import rgb2lab, lab2rgb, rgb2gray
import numpy as np
import os
from cv2 import imread, imshow, destroyAllWindows, waitKey, resize


# %%

processed_images_X_List_animals = []
processed_images_Y_List_animals = []
processed_images_X_List_nature = []
processed_images_Y_List_nature = []
processed_images_X_List_humans = []
processed_images_Y_List_humans = []
processed_images_X_List = []
processed_images_Y_List = []

# %%
def load_data(data_dir, size):
    count = 0
    for filename in os.listdir(data_dir):
        image = imread(os.path.join(data_dir, filename))
        image = resize(image, dsize=size)
        image = np.array(image, dtype=float)
        X = rgb2lab(1.0/255*image)[:,:,0]
        Y = rgb2lab(1.0/255*image)[:,:,1:]
        Y /= 128
        processed_images_X_List_humans.append(X)
        processed_images_Y_List_humans.append(Y)
        count += 1
        print(count, " - ", filename)
    
    
    print("Dosyalar Yüklendi")


load_data("F:\\Restoresyon\\data\\humans", (400,400))
# %%
print(len(processed_images_X_List_animals))
print(len(processed_images_X_List_nature))
print(len(processed_images_X_List_humans))

# %%
processed_images_X = np.array(processed_images_X_List_nature)
processed_images_Y = np.array(processed_images_Y_List_nature)

# %%
processed_images_X = np.concatenate([processed_images_X, np.array(processed_images_X_List_nature)])
processed_images_Y = np.concatenate([processed_images_Y, np.array(processed_images_Y_List_nature)])

# %%
processed_images_X = processed_images_X.reshape(-1, 400, 400, 1)
processed_images_Y = processed_images_Y.reshape(-1, 400, 400, 2)

def convert_to_lab(image1):
        
    X = rgb2lab(1.0/255*image1)[:,:,0]
    Y = rgb2lab(1.0/255*image1)[:,:,1:]
    Y /= 128
    X = X.reshape(1, 400, 400, 1)
    Y = Y.reshape(1, 400, 400, 2)

    return image1

# %%
# Building the neural network
model = Sequential()
model.add(InputLayer(shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

# %%
# Finish model
model.compile(optimizer='rmsprop',loss='mse')

# %%
model.fit(x=processed_images_X, 
    y=processed_images_Y,
    batch_size=1,
    epochs=50)

# %%

def color_me(image1):
    image1 = resize(image1, (400,400))
    image1 = np.array(image1, dtype=float)
    X = rgb2lab(1.0/255*image1)[:,:,0]
    
    X = X.reshape(1, 400, 400, 1)
    

    output = model.predict(X)
    output = output*128
    cur = np.zeros((400, 400, 3))
    cur[:,:,0] = X[0][:,:,0]
    cur[:,:,1:] = output[0]

    return cur

def color_me2(image1):
    original_shape = image1.shape[:2]  # Orijinal görüntünün genişlik ve yükseklik boyutları alınıyor
    #image1 = resize(image1, (400, 400))  # Görüntüyü yeniden boyutlandırma
    image1 = np.array(image1, dtype=float)  # Görüntüyü float tipine dönüştürme
    X = rgb2lab(1.0/255*image1)[:,:,0]  # Gri tonlamalı (L) kanalı alınıyor
    
    X = X.reshape(1, original_shape[0], original_shape[1], 1)  # Modelin beklentisi olan şekle dönüştürme
    

    output = model.predict(X)  # Modeli kullanarak renklendirme yapma
    output = output * 128  # Sonuçları uygun aralığa dönüştürme
    cur = np.zeros((original_shape[0], original_shape[1], 3))  # Renklendirilmiş görüntüyü tutacak matris oluşturma
    cur[:original_shape[0], :original_shape[1], 0] = X[0, :, :, 0]  # Orijinal boyutta L (parlaklık) kanalını yerleştirme
    cur[:original_shape[0], :original_shape[1], 1:] = output[0]  # Orijinal boyutta a* ve b* kanallarını yerleştirme

    return cur





# %%
#print(model.evaluate(processed_images_X, processed_images_Y, batch_size=1))

get_image = imread("test3.jpg")
#get_image = resize(get_image, (get_image.shape[1]//3, get_image.shape[0]//3))
cur = color_me2(get_image)
image = lab2rgb(cur)

#imagee = rgb2gray(lab2rgb(cur))
imshow("aa", image)
imshow("aaa", get_image)
waitKey()
destroyAllWindows()
#imsave("img_result.png", lab2rgb(cur))
#imsave("img_gray_version.png", rgb2gray(lab2rgb(cur)))

# %%



