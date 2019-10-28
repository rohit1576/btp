from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import cv2
from keras.models import load_model
from scipy import ndimage, misc

import pickle


from PIL import Image
img = Image.open('my_file.png')


model = pickle.load(open('finalized_model.sav', 'rb'))
ans = model.predict(img)
print(ans)
