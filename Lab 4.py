#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:27:19 2020

@author: user
"""

# import keras
# keras.__version__

#Importando la base de datos de numeros escritos a mano
from keras.datasets import mnist

#Carga a la base de datos de entrenamiento y testeo
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Son 60000 imagenes con una resolución de 28*28 pixeles
train_images.shape

#Con sus correspondientes 60000 etiquetas
len(train_labels)

print(train_labels.shape,train_labels[0])

#Por otra parte el conjunto de testeo

#Tenemos 10000 imagenes de 28x28 pixeles
test_images.shape

#10000 etiquetas
len(test_labels)

test_labels


import numpy as np
import matplotlib.pyplot as plt

img = np.random.randint(0, len(train_images))
digit = train_images[img]
plt.imshow(digit)
plt.show()
print('The label for this image is', train_labels[img])

#Las diferentes etiquetas
np.unique(train_labels)


#Lets train our ANN

from keras import models
from keras import layers

network =models.Sequential()
network.add(layers.Dense(512,activation="sigmoid",input_shape=(28*28,)))
network.add(layers.Dense(10,activation="sigmoid"))


#Lo que nos falta para poder compilar la red es:
#Una función de perdida
#Un metodo de optimización
#Una metríca de ajuste de la red

network.compile(optimizer="sgd",loss="categorical_crossentropy",
                metrics=["accuracy"])

#Preprocesando las imagenes para llevarlas al formato que espera la red


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print(train_labels)
#Al parecer son vector indicadores para la etiqueta

#Ahora procedemos a ajustar nuestro modelo a los datos


network.fit(train_images,train_labels,epochs=6,batch_size=128)

#Veamos que tal le va con nuestro conjunto de validación

test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc, '\ntest_loss:', test_loss)

#Veamos que tal nos fue con el conjunto de entrenamiento
train_loss, train_acc = network.evaluate(train_images, train_labels)

print('train_acc:', train_acc, '\ntrain_loss:', train_loss)

#Valores predichos

y_hat=network.predict_classes(train_images)

#Veamos la matriz de confusión de la red
import pandas as pd

pd.crosstab(train_labels,y_hat)


#Chekeando la predicción de las imagenes

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Sacamos una imagen aleatoria
img = np.random.randint(0, len(test_images))

#aplanando la imagen en una sola fila
check_image = test_images[img].reshape((1,28*28))

#Haciendo la predicción de clase
plt.imshow(test_images[img])
plt.show()
check_class = network.predict_classes(check_image)
print("El número predicho para la imagen obtenida es: {0}".format(check_class[0]))

prediction = check_class[0]

# =============================================================================
# Practica, modificando hiperparametros
# =============================================================================

#Preprocesando las imagenes para llevarlas al formato que espera la red


train_images_1 = train_images.reshape((60000, 28 * 28))
train_images_1 = train_images_1.astype('float32') / 255

test_images_1 = test_images.reshape((10000, 28 * 28))
test_images_1 = test_images_1.astype('float32') / 255

train_labels_1 = to_categorical(train_labels)
test_labels_1 = to_categorical(test_labels)

#Numero de epocas 
ann_dic={}
for k in [4,6,8,12]:

    network_epoch= models.Sequential()
    network_epoch.add(layers.Dense(512,activation="sigmoid",input_shape=(28*28,)))
    network_epoch.add(layers.Dense(10,activation="sigmoid"))

    network_epoch.compile(optimizer="sgd",loss="categorical_crossentropy",
                metrics=["accuracy"])    

    #Ajustando los datos a la red

    network_epoch.fit(train_images_1,train_labels_1,epochs=k,batch_size=128)
    
    ann_dic[k]=network_epoch.evaluate(train_images_1,train_labels_1)
    
    
#gráficado los resultados
x =[i for i in ann_dic]
y = [ann_dic[i][1] for i in ann_dic]


fig,ax = plt.subplots(1,1,figsize=(10,6))
ax.plot(x,y,"b",label="Relación")
#Conclusión al aumentar la cantidad de epocas en el algoritmo
#Parece haber un incremento en la presición del modelo
#Sin embargo esto tendrá una asintota a medida que crezcan la epocas


#Numero de batch
ann_dic={}
for k in [15,30,80,128,180,300]:

    network_epoch= models.Sequential()
    network_epoch.add(layers.Dense(512,activation="sigmoid",input_shape=(28*28,)))
    network_epoch.add(layers.Dense(10,activation="sigmoid"))

    network_epoch.compile(optimizer="sgd",loss="categorical_crossentropy",
                metrics=["accuracy"])    

    #Ajustando los datos a la red

    network_epoch.fit(train_images_1,train_labels_1,epochs=4,batch_size=k)
    
    ann_dic[k]=network_epoch.evaluate(train_images_1,train_labels_1)
    
     
#gráficado los resultados
x =[i for i in ann_dic]
y = [ann_dic[i][1] for i in ann_dic]


fig,ax = plt.subplots(1,1,figsize=(10,6))
ax.plot(x,y,"b",label="Relación")   
#Conclusión a menor número de batch_size, mejor presición se tiene sin embargo el ajuste se demora 
#mucho mas tiempo comparado con los demás





#Numero de neuronas por capa


ann_dic={}
for k in [150,300,512,750,1000]:

    network_epoch= models.Sequential()
    network_epoch.add(layers.Dense(k,activation="sigmoid",input_shape=(28*28,)))
    network_epoch.add(layers.Dense(10,activation="sigmoid"))

    network_epoch.compile(optimizer="sgd",loss="categorical_crossentropy",
                metrics=["accuracy"])    

    #Ajustando los datos a la red

    network_epoch.fit(train_images_1,train_labels_1,epochs=4,batch_size=128)
    
    ann_dic[k]=network_epoch.evaluate(train_images_1,train_labels_1)
    
     
#gráficado los resultados
x =[i for i in ann_dic]
y =[ann_dic[i][1] for i in ann_dic]


fig,ax = plt.subplots(1,1,figsize=(10,6))
ax.plot(x,y,"b",label="Relación")   

#Conclusión a mayor numero de neuronas en la capa se tiene mejor ajuste de los datos
#sin embargo tambien se tiene una asintota y se ve una decaida en masomenos 512







#Numero de capas


ann_dic={}
for k in [1,2,3,4,5,6,8]:

    network_epoch= models.Sequential()
    for i in range(k):
        network_epoch.add(layers.Dense(512,activation="sigmoid",input_shape=(28*28,)))
    
    network_epoch.add(layers.Dense(10,activation="sigmoid"))

    network_epoch.compile(optimizer="sgd",loss="categorical_crossentropy",
                metrics=["accuracy"])    

    #Ajustando los datos a la red

    network_epoch.fit(train_images_1,train_labels_1,epochs=4,batch_size=128)
    
    ann_dic[k]=network_epoch.evaluate(train_images_1,train_labels_1)
    
     
#gráficado los resultados
x =[i for i in ann_dic]
y =[ann_dic[i][1] for i in ann_dic]


fig,ax = plt.subplots(1,1,figsize=(10,6))
ax.plot(x,y,"b",label="Relación")   

#Conclusión a mayor numero de neuronas en la capa se tiene mejor ajuste de los datos
#sin embargo tambien se tiene una asintota y se ve una decaida en masomenos 512

