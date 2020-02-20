#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 22:39:56 2020

@author: user
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns

#from subprocess import check_output

# =============================================================================
# Preparing de dataset
# =============================================================================

from keras.datasets import mnist

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

#Shuffle training and test set

#Variable con los indices del arreglo
s = np.arange(Y_test.shape[0])

np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s]

p = np.arange(Y_train.shape[0])
np.random.shuffle(p)
X_train = X_train[p]
Y_train = Y_train[p]

#Se usará solo un subconjunto de todos los datos para este ejercicio
X_train = X_train[0:6000]
Y_train = Y_train[0:6000]

X_test = X_test[500:1000]
Y_test = Y_test[500:1000]

#Redimensionando los arreglos

#Originales
X_train.shape
X_train = X_train.reshape(len(X_train),28,28,1)
X_train.shape

X_test.shape
X_test = X_test.reshape(len(X_test), 28, 28, 1)
X_test.shape

#Normalización de los datos y balanceo de las clases

# normalization
X_train = np.array(X_train) / 255.
Y_train = np.array(Y_train)
X_test = np.array(X_test) / 255.
Y_test = np.array(Y_test)
#X_valid = np.array(X_valid) / 255.
#Y_valid = np.array(Y_valid)

print('There are', X_train.shape[0], 'training data and',  X_test.shape[0], 'testing data.')
print('Number of occurence for each number in training data (0 stands for 10):')
print(np.vstack((np.unique(Y_train), np.bincount(Y_train))).T)

fig,ax = plt.subplots(1,1,figsize=(7,8))
sns.countplot(Y_train)
ax.set_title("Conteo de las diferentes etiquetas")
ax.set_ylabel("Conteo")
ax.set_xlabel("Categoría")
ax.grid(True,lw=0.75,ls="--",alpha=0.75)

#Plotting first 36 images in MNIST

# plot first 36 images in MNIST
fig, ax = plt.subplots(6, 6, figsize = (12, 12))
fig.suptitle('First 36 images in MNIST')
fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
for x, y in [(i, j) for i in range(6) for j in range(6)]:
    ax[x, y].imshow(X_train[x + y * 6].reshape((28, 28)), cmap = 'gray')
    ax[x, y].set_title(Y_train[x + y * 6])
    ax[x,y].get_xaxis().set_visible(False)
    ax[x,y].get_yaxis().set_visible(False)

#Vectorizando las etiquetas de las imagenes
    
from keras.utils import to_categorical

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

 # Notice that the first dimension is the number of examples in 
# the training set, the second and third are the dimensions of each 
# image, and the third is the number of 'colors' in which the images 
# are available. In this case we only have one color, so the dimension
 # is one
print(np.shape(X_train))

# =============================================================================
# Building the CNN Model
# =============================================================================

#Implementaremos dos capas con Convolución

# Import needed models and layers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

img_size = (28,28,1) # Dimensions of the input volume
n_classes = 10       # Number of classes
deep_c1=32
deep_c2=32
#Modelo secuencial inicial
model = Sequential()

#Primera capa convolucional
model.add(Conv2D(deep_c1, (5, 5), input_shape = img_size, kernel_initializer = 'normal'))
model.add(Activation('relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))#pool_size= submatriz a analizar

#Segunda Capa convolucional
model.add(Conv2D(deep_c2, (5, 5), kernel_initializer = 'normal'))
model.add(Activation('relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))#pool_size= submatriz a analizar
model.add(Dropout(0.05))#Probabilidad de desactivar un filtro de la capa
model.add(Flatten())

#Capas Fully Connected
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(n_classes))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])


history = model.fit(X_train, Y_train, batch_size = 128, epochs = 5, 
          validation_split = 0.2, verbose = 1)

model.summary()


#Gráficando el ajuste

# Plot: Loss History during Training and Receiver-Operator Curve (ROC)

try:
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
except: 
    pass
hist = history.history
acc = hist['acc']
val_acc = hist['val_acc']
epochs = np.arange(5)
fig, axis1 = plt.subplots(figsize=(7,9))
plt.plot(epochs+[1]*5, acc, 'b', label='acc')
plt.plot(epochs+[1]*5, val_acc, 'r', label="val acc")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Accurracy History")
plt.legend(loc='upper right')
plt.tight_layout()

# Testing
score, acc = model.evaluate(X_test, Y_test, verbose = 1)
print(score)
print(acc)


Y_pred = model.predict(X_test)
Y_pred_real = []
Y_test_real = []
Y_prob = []

for i in range(len(Y_pred)):
    Y_pred_real.append(np.argmax(Y_pred[i], axis=None, out=None))
    Y_test_real.append(np.argmax(Y_test[i], axis=None, out=None))
    Y_prob.append(np.max(Y_pred[i]))
print(Y_pred_real)
print(Y_test_real)

from sklearn.metrics import confusion_matrix
from sklearn import metrics
# Get confusion matrix
cm = confusion_matrix(Y_test_real, Y_pred_real)
cm

#Normalize it
cm_norm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]


# Ploteando la matriz de confusión de manera proporcional por fila
plt.imshow(cm_norm, cmap = 'gray')
plt.title('Normalized confusion matrix')
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, np.arange(n_classes), rotation=0)
plt.yticks(tick_marks, np.arange(n_classes))
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Visualizating filters
# https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
from keras import backend as K
K.set_learning_phase(1)
import tensorflow as tf

#Extraer las capas del modelo
layer_dict = dict([(layer.name, layer) for layer in model.layers])
#print('Layer dict', layer_dict)
print(model.summary())

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean() # x=x-x.mean()
    x /= (x.std() + 1e-5)#Porque sumar un positivo pequeño??
    x *= 0.1

    # clip to [0, 1]
    x += 0.5 #Muy parecido a lo de autoencoders
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#Obtiene todas las salidas de la segunda capa convolucional
layer_output = layer_dict['conv2d_17'].output

print (layer_output)



def vis_img_in_filter(img = np.array(X_train[1]).reshape((1, 28, 28, 1)).astype(np.float64), 
                      layer_name = 'conv2d_17'):
    layer_output = layer_dict[layer_name].output
    img_ascs = list()
    for filter_index in range(layer_output.shape[3]):
        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        loss = K.mean(layer_output[:, :, :, filter_index])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, model.input)[0]

        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([model.input], [loss, grads])

        # step size for gradient ascent
        step = 5.

        img_asc = np.array(img)
        # run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([img_asc])
            img_asc += grads_value * step

        img_asc = img_asc[0]
        img_ascs.append(deprocess_image(img_asc).reshape((28, 28)))
        
    if layer_output.shape[3] >= 35:
        plot_x, plot_y = 6, 6
    elif layer_output.shape[3] >= 23:
        plot_x, plot_y = 4, 6
    elif layer_output.shape[3] >= 11:
        plot_x, plot_y = 2, 6
    else:
        plot_x, plot_y = 1, 2
    fig, ax = plt.subplots(plot_x, plot_y, figsize = (12, 12))
    ax[0, 0].imshow(img.reshape((28, 28)), cmap = 'gray')
    ax[0, 0].set_title('Input image')
    ax[0, 0].get_xaxis().set_visible(False)
    ax[0, 0].get_yaxis().set_visible(False)
    fig.suptitle('Input image and %s filters' % (layer_name,))
    fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
    for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
        if x == 0 and y == 0:
            continue
        ax[x, y].imshow(img_ascs[x * plot_y + y - 1], cmap = 'gray')
        ax[x, y].set_title('filter %d' % (x * plot_y + y - 1))
        ax[x, y].get_xaxis().set_visible(False)
        ax[x, y].get_yaxis().set_visible(False)

vis_img_in_filter()


#Nota: al parecer este metodo cuenta con un factor aleatorio que afecta
#mucho el ajuste del modelo

#El mejor ajuste se obtuvo con disminuyendo en # de filtros en cada
#capa convolucional, además de reducir el factor del Dropout 
#del 10% al 5% 