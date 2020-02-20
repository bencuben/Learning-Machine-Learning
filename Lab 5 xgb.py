#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:07:42 2020

@author: user
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
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

model_gbt = XGBClassifier(objective="multi:softmax",learning_rate=0.1,
                          n_estimators=300,
                          max_depth=8,min_child_weight=8,
                          nthread=4,subsample=1,colsample_bytree=0.6)


model_gbt.fit(X=X_train,y=y_train.values.ravel(),eval_metric="merror")



# Encontrar importancia de cada variable, y graficar
importanciaVars=model_gbt.feature_importances_

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

y_pred = model_gbt.predict(X_test)
precision=accuracy_score(y_test, y_pred)
print(precision)

# Matriz de confusion
tabla=pd.crosstab(y_test.values.ravel(), y_pred, rownames=['Actual LOS'], colnames=['Predicted LOS'])
print(tabla)

