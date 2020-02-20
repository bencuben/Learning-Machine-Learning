#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 18:17:54 2020

@author: user
"""



#Carga de paquetes necesarios 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Carga inicial de datos
X = pd.read_csv("/home/user/anaconda3/lml2018/Labs/Lab5/data_preprocesada.csv")

#Veamos sus dimensiones
X.shape

#Cargando la variable respuesta
y= pd.read_csv("/home/user/anaconda3/lml2018/Labs/Lab5/y.csv")

#Nombramiento de las variables 
feat_label = ['Diagnostico','Hospital','via_Ingreso','codigo_Administradora','Causa_Externa','Edad','Ocupacion','Num_Reinserciones']


# Separar en datos de entrenamiento y validacion
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Entrenar el clasificador

#n_estimators=Número de arboles en el bosque???
#max_features= numero de variables para cada arbol
#random_state= Inicializador del aleatorio????

model_rf =RandomForestClassifier(n_estimators=500,max_features=4,
                                 min_samples_leaf=10,random_state=0,
                                 n_jobs=2)
 
#Se ajustan los datos al modelo
model_rf.fit(X_train,y_train.values.ravel())

# Encontrar importancia de cada variable, y graficar
importanciaVars=model_rf.feature_importances_

# Graficar la importancia de las variables
pos=[1, 2, 3, 4, 5, 6, 7, 8]
plt.rcdefaults()
fig, ax = plt.subplots(1,1,figsize=(10,6))
ax.barh(pos, importanciaVars, align='center',color='blue')
ax.set_yticks(pos)
ax.set_yticklabels(feat_label)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Importancia Variables')
plt.show()


# Realizar prediccion en datos de validacion
y_pred = model_rf.predict(X_test)
precision=accuracy_score(y_test, y_pred)
print(precision)

# Matriz de confusion
tabla=pd.crosstab(y_test.values.ravel(), y_pred, rownames=['Actual LOS'], colnames=['Predicted LOS'])
print(tabla)
