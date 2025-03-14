import numpy as np
import cv2

from keras.models import load_model

def loadModel():
    return load_model("model/generator_model_epoch50.h5")





def findsize(image):
    
    height, width = image.shape[:2]

    sizes = [256, 512, 1024, 2048, 4096]

    
    new_height = next((x for x in sizes if x >= height), sizes[-1])
    new_width = next((x for x in sizes if x >= width), sizes[-1])

    return new_height, new_width


def prepare(image):

    height, widt = image.shape[:2]    

    img = np.copy(image)

    if len(image.shape) < 3:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    new_height, new_width = findsize(image)
    
    img = cv2.resize(img, (new_height, new_width))

    img = img.astype("float32") / 255.0

    img = img.reshape(-1, new_height, new_width, 3)

    return img


def colorize(image):

    height, width = image.shape[:2]

    # pix, color = Pixels(image)

    img = prepare(image)
    # pix_pos, values = convert_and_extract_pixels(img)

    model = loadModel()

    prediction = model(img, training=True)

    
    img = (np.array(prediction[0])*255).astype('uint8')


    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)


    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    return img
    
