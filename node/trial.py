import string
import random
from random import randint
import cv2
import numpy as np
import scipy

import math
import numpy as np
#from ipywidgets import interact
#import ipywidgets as ipywidgets
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
from os import listdir
from os.path import isfile, join

def files_in_folder(folder_path):
  '''
  Returns a list of strings where each entry is a file in the folder_path.
  
  Parameters
  ----------
  
  folder_path : str
     A string to folder for which the file listing is returned.
     
  '''
  files_A = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
  print(files_A)
  # The files when listed from Google Drive have a particular format. They are
  # grouped in sets of 4 and have spaces and tabs as delimiters.
  
  # Split the string listing sets of 4 files by tab and space and remove any 
  # empty splits.
  f#iles_B = [list(filter(None, re.split('\t|\s', files))) for files in files_A]
  
  # Concatenate all splits into a single sorted list
  files_C = []
  #for element in files_B:
    #files_C = files_C + element
  #files_C.sort()
  
  return files_A

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def reset_weights(model):
    session = backend.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

PATH = "/home/fizzer/ros_ws/src/controller_pkg/node/letters"

alllets = []
labels = []
folder1 = PATH

files1 = files_in_folder(folder1)
#print(files1)


for file in files1:
  pic = cv2.imread(PATH+"/"+file, cv2.IMREAD_UNCHANGED)
  alllets.append(pic)
  labels.append(np.eye(26)[ord(file[0])-65])

X_dataset_orig = np.array(alllets)
Y_dataset= np.array(labels)
shape = X_dataset_orig.shape
X_dataset = X_dataset_orig.reshape((shape[0],shape[1],shape[2],1))

VALIDATION_SPLIT = 0.2

conv_model = models.Sequential()
conv_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                             input_shape=(45, 35,1)))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#conv_model.add(layers.MaxPooling2D((2, 2)))
#conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Flatten())
conv_model.add(layers.Dropout(0.5))
conv_model.add(layers.Dense(512, activation='relu'))
conv_model.add(layers.Dense(26, activation='softmax'))

LEARNING_RATE = 1e-4
conv_model.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                   metrics=['acc'])
reset_weights(conv_model)


Xtrainlist = []
Ytrainlist = []

Xvallist = []
Yvallist = []

shape = X_dataset.shape

valSize = shape[0]*0.2
valIndex = np.random.randint(0,int(shape[0]),int(valSize))

for i in range(shape[0]):
  if i in valIndex:
    Xvallist.append(X_dataset[i])
    Yvallist.append(Y_dataset[i])
  else:
    Xtrainlist.append(X_dataset[i])
    Ytrainlist.append(Y_dataset[i])

X_train = np.array(Xtrainlist)
Y_train = np.array(Ytrainlist)

X_val = np.array(Xvallist)
Y_val = np.array(Yvallist)

print("It's Train Time!")

history_conv = conv_model.fit(X_dataset, Y_dataset, 
                              validation_data=(X_val, Y_val), 
                              epochs=45, 
                              batch_size=16)

print("And we done!")

conv_model.save_weights('/home/fizzer/ros_ws/src/controller_pkg/node/letterweights')
conv_model.save('/home/fizzer/ros_ws/src/controller_pkg/node/my_model_letters.h5')



