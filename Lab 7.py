#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 21:33:18 2020

@author: user
"""

#Neural networks part 2

#Validación del modelo: veremos 3 principales métodos.
#hold-put,k-fold,iterated k-fold with shuffling.



# =============================================================================
# Simple hold-out validation
# =============================================================================

#Este metodo consiste en la partición del dataset en entrenamiento y validación
#el problema es que si se tiene "pocos" datos, el ajuste cambiara mucho de
#muestra a muestra

# =============================================================================
# K-fold validation
# =============================================================================


#Se parte el dataset en k partes iguales
# En este método para cada partición i= 1,2,...k.
# Se entrena el modelo con el complemento y se valida con la partición i-ésima.

#Luego se promedia los scores obtenidos

#Recordar que siempre se debe tomar un conjunto de datos diferente
#Sea un dataset 2 para la calibración del modelo.

# =============================================================================
# Iterated k-fold with shuffling
# =============================================================================

#Para relativamente "pocos" datos 
#Consiste en aplicar P veces el método de k-fold donde antes de partir
# el data set en k partes, se ordena alteatoriamente.

#El score final es el promedio de los promedio obtenidos en cada iteración.



# =============================================================================
# Overfitting and Underfitting
# =============================================================================

#Primeramente en los métodos de machine learning es importante el tamaño
#grande de muestra para evitar el sobre ajuste

#El proceso del intercambio entre ajuste y generalización lo llaman
#REGULARIZACIÓN

# =============================================================================
#CÓDIGO 
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Definiendo funciones auxiliares

#Función de ajuste de Red Neuronal con una capa oculta 
def Train(training_data, training_labels, validation_data, validation_labels):
    """Insert docstring here..."""
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                    metrics=['accuracy'])

    network.fit(training_data, training_labels, epochs=5, batch_size=128)

    training_score = network.evaluate(training_data, training_labels)
    validation_score = network.evaluate(validation_data, validation_labels)
    
    return validation_score, training_score

def Prepare(imgs, labs):
    """Insert docstring here..."""
    p_imgs = imgs.reshape((len(imgs), 28 * 28))
    p_imgs = p_imgs.astype('float32') / 255
    p_labs = to_categorical(labs)
    
    return p_imgs, p_labs


#Evluación del modelo con k-fold
    
k = 5
split = len(train_images) // k
print('split', split)
val_scores, tra_scores = [], []

for fold in range(k):
    print('*'*40,"\n", 'fold', fold,"\n", '*'*40)
    beg, end = fold * split, (fold + 1) * split    
    #print(beg, end)
    validation_data = train_images[beg:end, :, :]
    training_data = np.concatenate((train_images[:beg, :, :], train_images[end:, :, :]))
    
    validation_labels = train_labels[beg:end]
    training_labels = np.concatenate((train_labels[:beg], train_labels[end:]))
    #print(validation_labels.shape, training_labels.shape)
    
    tra_data, tra_lab = Prepare(training_data, training_labels)
    val_data, val_lab = Prepare(validation_data, validation_labels)
    
    val_score, tra_score = Train(tra_data, tra_lab, val_data, val_lab)
    val_scores.append(val_score)
    tra_scores.append(tra_score)
    

list(zip([i+1 for i in range(k)],val_scores ) )
list(zip([i+1 for i in range(k)],tra_scores ) )
#Calculando los scores promedios

avg_tra_score = np.average(tra_scores, axis=0)
print('\navg_tra_score', avg_tra_score)

avg_val_score = np.average(val_scores, axis=0)
print( '\navg_val_score', avg_val_score)

#Visualización de resultados

fig,ax = plt.subplots(1,1,figsize=(10,6))
ax.set_ylim(0.95,1)
ax.plot([1,2,3,4,5],np.array(tra_scores)[:,1], '-o', label='training')
ax.plot([1,2,3,4,5],np.array(val_scores)[:,1], '--s', label='validation')
ax.set_ylabel('accuracy')
ax.set_xlabel('folding')
ax.set_title("Precisión en cada una de las particiones")
ax.legend(loc="lower left")

# =============================================================================
# FUnción para validación
# =============================================================================

#Veremos que la opción model.fit contiene un argumento "validation_split"
#Que nos será de utilidad

# validation_split: Float between 0 and 1. Fraction of the training
# data to be used as validation data. The model will set apart this 
#fraction of the training data, will not train on it, and will 
#evaluate the loss and any model metrics on this data at the end 
#of each epoch. The validation data is selected from the last samples 
#in the x and y data provided, before shuffling.

#Procesamiento de las imagenes
train_data, train_lab = Prepare(train_images, train_labels)

#Ajuste del modelo
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])

r = network.fit(train_data, train_lab, validation_split=0.333333, 
                epochs=5, batch_size=128)

train_loss, train_acc = network.evaluate(train_data, train_lab)

#Recolentado la información del historico
r_acc = r.history['acc']
v_acc = r.history['val_acc']

#Gráficando el ajuste
fig,ax = plt.subplots(1,1,figsize=(10,6))
ax.set_ylim(0.90,1)
ax.plot([1,2,3,4,5],r_acc,'-o', label='training')
#NOTA: "-o" es igual a marker='o',linestyle='-'
ax.plot([1,2,3,4,5],v_acc, '--s', label='validation')
#NOTA: "-s" es igual a marker='s',linestyle='--'
ax.set_ylabel('accuracy')
ax.set_xlabel('epochs')
ax.set_title("Contraste de ajustes según la epoca")
ax.legend(loc="upper left")



# =============================================================================
# EJERCICIO IMPLEMENTAR EL iterated k-fold with suffle
# =============================================================================

#Conversión de las imagenes al formato necesitado por la red
tra_data, tra_lab = Prepare(train_images, train_labels)

#Definición de las variables del método

#Numero de veces que se mezclan los datos
p = 3
#Número de particiones en cada mezcla
k = 5
#Tamaño de las particiones
split = len(train_images) // k
#Lista para almacenar resultados de cada iteración del K-fold
val_scores, tra_scores = [], []
#Lista para almacenar los promedios de las p iteraciones del k-fold
avg_tra_score,avg_val_score = [],[]


for j in range(p):        
    
    #Reordenamos todos los indices
    orden = np.random.choice(list(range(60000)),size=60000,replace=False)
    
    #Y creamos unas bases auxiliares con este reordenamiento para el entrenamiento
    t_data=tra_data[orden,:]
    t_label=tra_lab[orden,:]
    
    for fold in range(k):
        
        #Particionando los datos de acuerdo al K-fold
        print('*'*40,"\n","P = ",j+1, ', fold', fold+1 ,"\n", '*'*40)
        beg, end = fold * split, (fold + 1) * split    
        
        #División para la i-ésima partición 
        validation_data = t_data[beg:end, :]
        validation_labels = t_label[beg:end,:]
        
        training_data = np.concatenate((t_data[:beg, :], t_data[end:, :]))      
        training_labels = np.concatenate((t_label[:beg,:], t_label[end:,:]))
        
    
        #Entrenamiento del modelo con la función previamente definida
        val_score, tra_score = Train(training_data,
                                     training_labels,
                                     validation_data,
                                     validation_labels)
        #Scores del modelo en testeo y entrenamiento
        val_scores.append(val_score)
        tra_scores.append(tra_score)
        #Fin primer For
    
    #Después de terminar cada vuelta del K-fold promediamos sus resultados    
    avg_tra_score.append(np.average(tra_scores, axis=0) )
    avg_val_score.append( np.average(val_scores, axis=0) )

#MEDIDAS SOBRE EL CONJUNTO DE VALIDACIÓN
resultados = [[0.08305493, 0.976225  ],[0.08296951, 0.97615556],[0.08310098, 0.9760625 ]]

resultados = [i[1] for i in resultados]

#Graficando los resultados
fig,ax = plt.subplots(1,1,figsize=(7,9))
ax.set_ylim(0.97,0.98)
ax.plot([1,2,3],resultados,"-o",label="Validación")
ax.set_title("Iterated k-fold with shuffle")
ax.set_xlabel("p")
ax.set_ylabel("Accuracy")

print("El promedio de las p iteracion del k-fold es {}".format(np.average(resultados)))
