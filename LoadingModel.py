from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import cv2
from keras.models import load_model
from scipy import ndimage, misc
from sklearn import metrics
import numpy
import pickle


from PIL import Image
img = Image.open('my_file.png')

img_width, img_height = 150 , 150
train_data_dir = 'dataset1/test_set'
validation_data_dir = 'dataset1/training_set'


model = pickle.load(open('finalized_model.sav', 'rb'))

#-------------------------------------------------------------

test_generator = ImageDataGenerator()
test_data_generator = test_generator.flow_from_directory(
                                                         validation_data_dir, # Put your path here
                                                         target_size=(img_width, img_height),
                                                         batch_size=32,
                                                         shuffle=False)
test_steps_per_epoch = numpy.math.ceil(test_data_generator.samples / test_data_generator.batch_size)

predictions = model.predict_generator(test_data_generator, steps=test_steps_per_epoch)
# Get most likely class
predicted_classes = numpy.argmax(predictions, axis=1)


true_classes = test_data_generator.classes
class_labels = list(test_data_generator.class_indices.keys())

report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)




#--------------------------------------------------------------

