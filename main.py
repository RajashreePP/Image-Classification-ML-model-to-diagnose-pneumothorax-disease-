import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Random Data Augmentation(Rescale, Rotation, Flips, Zoom, Shifts) using ImageDataGenerator 
training_data_generator = ImageDataGenerator(
    rescale = 1.0/255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

# Image Directory
training_image_directory = "/content/PRO-M3-Pneumothorax-Image-Dataset/training_dataset"

# Generate Preprocessed Augmented Data
training_augmented_images = training_data_generator.flow_from_directory(
    training_image_directory,
    target_size=(180,180))

# Random Data Augmentation(Rescale) using ImageDataGenerator
validation_data_generator = ImageDataGenerator(rescale = 1.0/255)

# Image Directory
validation_image_directory = "/content/PRO-M3-Pneumothorax-Image-Dataset/validation_dataset"

# Generate Preprocessed Augmented Data
validation_augmented_images = validation_data_generator.flow_from_directory(
    validation_image_directory,
    target_size=(180,180))

# Class Labels

training_augmented_images.class_indices

# Define/Build Convolution Neural Network

import tensorflow as tf
model = tf.keras.models.Sequential([
    
    # 1st Convolution & Pooling layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # 2nd Convolution & Pooling layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # 3rd Convolution & Pooling layer
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # 4th Convolution & Pooling layer
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a Dense Layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    # Classification Layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

# Model Summary

model.summary()

# Compile Model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit & Save Model

history = model.fit(training_augmented_images, epochs=20, validation_data = validation_augmented_images, verbose=True)

model.save("Pneumothorax.h5")

# Predict the Class of an Unseen Image
training_augmented_images.class_indices

import os
import numpy as np

from matplotlib import pyplot
from matplotlib.image import imread

import tensorflow
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# Testing image directory
testing_image_directory = '/content/PRO-M3-Pneumothorax-Image-Dataset/testing_dataset/infected'

# All image files in the directory
img_files = os.listdir(testing_image_directory)

i= 0

# Loop through an 9 image files
for file in img_files[51:60]:

  # full path of the image
  img_files_path = os.path.join(testing_image_directory, file)

  # load image 
  img_1 = load_img(img_files_path,target_size=(180, 180))

  # convert image to an array
  img_2 = img_to_array(img_1)

  # increase the dimension
  img_3 = np.expand_dims(img_2, axis=0)
  
  # predict the class of an unseen image
  prediction = model.predict(img_3)
  # print(prediction)

  predict_class = np.argmaxi(prediction, axis=1)
  # print(predict_class)

  # plot the image using subplot
  pyplot.subplot(3, 3, i+1)
  pyplot.imshow(img_2.astype('uint8'))
  
  # Add title of the plot as predicted class value
  pyplot.title(predict_class[0])

  # Do not show x and y axis with the image
  pyplot.axis('off')

  i=i+1

pyplot.show()

# Accuracy Curve

from matplotlib import pyplot

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# print(acc)
# print(val_acc)

epochs = range(len(acc))

pyplot.plot(epochs, acc, 'r', label='Training accuracy')
pyplot.plot(epochs, val_acc, 'b', label='Validation accuracy')

pyplot.title('Training and validation accuracy')

pyplot.legend()

pyplot.show()