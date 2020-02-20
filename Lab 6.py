#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 17:11:55 2020

@author: user
"""

#Auto encoders

#“Autoencoding” es un algoritmo de compresión de datos en donde cada 
#una de las funciones de compresión y descompresión

from keras.layers import Input,Dense
from keras.models import Model
from keras import optimizers

#Numero de neuronas para el compresor
encoding_dim = 32
#Dimensiones de las imagenes
input_img = Input(shape=(28*28,))

print(input_img)

encoded = Dense(encoding_dim, activation='relu')(input_img) 
decoded = Dense(784, activation='relu')(encoded)

#Modelos para la codificacion

autoencoder= Model(input_img,decoded)
encoder = Model(input_img,encoded)


#Modelo de decodificación

encoded_input = Input(shape=(encoding_dim,)) # Entrada codificada, de 32 dimensiones 
decoder_layer = autoencoder.layers[-1] # Traemos la última capa del modelo 
decoder = Model(encoded_input, decoder_layer(encoded_input)) # Creamos el modelo decodificado

#Selección de perdida y optimizador

# Configura el modelo para el entrenamiento, se puede seleccionar 
#el optimizador, la funcion de pérdida, la métricas, tensores objetivo,
# etc

autoencoder.compile(optimizer="adadelta",loss="mean_squared_error")


#Datos

from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

print(x_train[1])

#Llevamos los número que imagino que identidican las caracterisitcas de
#la imagen a un rango 0-1 dividiendo por el máximo
x_train = x_train.astype('float32') / 255. 
x_test = x_test.astype('float32') / 255.

len(x_train)

#Se multiplican la cantidad de pixeles 28x28
np.prod(x_test.shape[1:])

#convirtiendo en arreglos 60 000 x 784
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

x_train.shape
x_test.shape


history = autoencoder.fit(x_train, x_train,
                epochs=30,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

#Calculando las predicciones
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

#Graficando los resultados

import matplotlib.pyplot as plt

n = 10  # Cuántos dígitos queremos codificar
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#En la primera fila se ubican las imagenes reales
#En la segunda las imagenes reconstruidas

# =============================================================================
# Eliminación de Ruido
# =============================================================================

#Un factor de ruido inicial para sumar a las entradas de la imagen
noise_factor = 0.5

x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0,scale =1.0,size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
#Veamos los rangos de la matriz
np.min(x_train_noisy),np.max(x_train_noisy)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
#Luego de los recortes
np.min(x_train_noisy),np.max(x_train_noisy)

#Grafica perturbada
n = 10  # Cuántos dígitos queremos graficar
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


encoding_dim = 32

input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)


encoder = Model(input_img, encoded)
autoencoder = Model(input_img, decoded)


encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1] 
decoder = Model(encoded_input, decoder_layer(encoded_input)) 
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error') 

#Ajuste del modelo
history = autoencoder.fit(x_train_noisy, x_train,
                epochs=30,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))


autoencoder.summary()


encoded_imgs = encoder.predict(x_test_noisy)
decoded_imgs = decoder.predict(encoded_imgs)
#Resumen de la red, de hecho vemos que la red tiene 25,120 parametros que es igual
#(784+1)*32 donde 784 es la cantidad de caracteristicas de entradas
#+1 que es el "sesgo"(bias) por cada 1 de los 32 neuronas entrenadas

encoder.summary()

import matplotlib.pyplot as plt

n = 10  # Cuántos dígitos queremos codificar
plt.figure(figsize=(30, 4))

#NOTA: las gráficas no son 0 indexadas
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display original
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# =============================================================================
# Nuevo optimizador
# =============================================================================




#Numero de neuronas para el compresor
encoding_dim = 32
#Dimensiones de las imagenes
input_img = Input(shape=(28*28,))

print(input_img)

encoded = Dense(encoding_dim, activation='relu')(input_img) 
decoded = Dense(784, activation='relu')(encoded)

#Modelos para la codificacion

autoencoder= Model(input_img,decoded)
encoder = Model(input_img,encoded)


#Modelo de decodificación

encoded_input = Input(shape=(encoding_dim,)) # Entrada codificada, de 32 dimensiones 
decoder_layer = autoencoder.layers[-1] # Traemos la última capa del modelo 
decoder = Model(encoded_input, decoder_layer(encoded_input)) # Creamos el modelo decodificador

#Selección de función de perdida y optimizador
sgd = optimizers.sgd(lr=0.01,decay=1e-6,
                    momentum=0.9,nesterov=True)

autoencoder.compile(loss="mean_squared_error",optimizer=sgd)

#Ajuste del modelo
history = autoencoder.fit(x_train_noisy, x_train,
                epochs=30,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))


autoencoder.summary()

#Gráficando el entrenamiento del modelo a través de las epocas
y = history.history["loss"]
x = [i+1 for i in range(len(y))]

fig,ax = plt.subplots(1,1,figsize=(10,6))
ax.plot(x,y,"o",label="Ajuste")
ax.set_xlabel("Epocas")
ax.set_ylabel("MSE")
ax.set_title("Ajuste del modelo")
ax.grid()

#Graficando el ajuste con el nuevo optimizador


#NOTA: las gráficas no son 0 indexadas
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display original
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

mse_train= autoencoder.evaluate(x_train_noisy, x_train)
mse_test = autoencoder.evaluate(x_test_noisy, x_test)
print(mse_train,mse_test)




