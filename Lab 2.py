lik#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 22:38:38 2020

@author: user
"""

#Nota para instalar paquetes se usa los comandos

#pip install -U scikit-learn

#Nota: esta es la libreria de machine learning
#import sklearn

#import numpy as np
#from sklearn.linear_model import LinearRegression

# =============================================================================
# Parte 1
# =============================================================================

#Regresion linear con un juguete

x,y =[1,2,3],[2,2,4]



#Intentando el primer ejemplo
import numpy as np
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt


#Variable dependiente
x=np.array(x)

#Variable independiente
y=np.array(y)

#Creando un objeto que contiene caracteristicas de modelo linear
model = LinearRegression()
#Ajustar el modelo con estos datos
model.fit(x[::,np.newaxis],y)
#Predigame con las variable de entrenamiento
y_test= model.predict(x[::,np.newaxis])

#Comparación gráfica
fig,ax =plt.subplots(1,1,figsize=(5,5))

ax.scatter(x,y,label="Real")
ax.set_ylim(1,5)
ax.legend(["linea básica"],loc="upper left")

ax.plot(x,y_test,ls="-.",label="prediccion")
ax.legend(loc="center left")


#Datos matriciales de laregresión linear
#ejemplo:

X1 = np.array([[1, 1, 1], [1, 2, 1], [1, 3, 1]])
print(X1)
#Para calcular la invertibilidad de una matriz se mira si el determinante es cero o no
print(np.linalg.det(np.dot(np.transpose(X1),X1))  )

#Ejemplo 2:
X2 = np.array([[1, 1, 1], [1, 2, 1], [1, 3, 4]])
print(X2)
print(np.linalg.det(np.dot(np.transpose(X2), X2)))


#BUILDING A MODEL FRIM SCRATCH

import numpy as np
import pandas as pd
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
import statsmodels.api as sm


#observed predictors
x_train =np.array([1,2,3])

# or do this, which creates 3 x 1 vector so no need to reshape
#x_train = np.array([[1], [2], [3]])   
print(x_train.shape)
print(x_train)

#Para cambiar las dimensiones se usa
x_train = x_train.reshape(3,1)
x_train


#observed responses
y_train = np.array([2, 2, 4])
# or do this, which creates 3 x 1 vector so no need to reshape
#y_train = np.array([[2], [2], [4]])
y_train = y_train.reshape(len(y_train),1)
print(y_train.shape)
print(y_train)

#build matrix X by concatenating predictors and a column of ones
n = x_train.shape[0]
#Creando un vector de unos compatible
ones_col = np.ones((n, 1),dtype=int)

#Concatenar por filas
X = np.concatenate((ones_col, x_train), axis=1)


#check X and dimensions
print(X, X.shape)

#Creando el vector de betas estimados
hat=np.dot(np.transpose(X),X)

betas= np.dot(np.dot(np.linalg.inv( hat ), np.transpose(X) ),y_train)


#### Manera del texto

#matrix X^T X
LHS = np.dot(np.transpose(X), X)
print(LHS)

#matrix X^T Y
RHS = np.dot(np.transpose(X), y_train)
print(RHS)

#solution beta to normal equations, since LHS is invertible by toy construction
betas = np.dot(np.linalg.inv(LHS), RHS)
print(betas)#Misma solución que antes

#intercept beta0
beta0 = betas[0]

#slope beta1
beta1 = betas[1]

print(beta0, beta1)





#    EXERCISE: Turn the code from the above cells into a function, called simple_linear_regression_fit, that inputs the training data and returns beta0 and beta1.
#
#    To do this, copy and paste the code from the above cells below and adjust the code as needed, so that the training data becomes the input and the betas become the output.
#
#    Check your function by calling it with the training data from above and printing out the beta values.
#
#    Then plot the trainig points with the fitted model








#your code here
def simple_linear_regression_fit(x_train, y_train):
    
    #reshape inputs 
    x_train = x_train.reshape(len(x_train),1)
    y_train = y_train.reshape(len(y_train),1)

    #build matrix X by concatenating predictors and a column of ones
    n = x_train.shape[0]
    ones_col = np.ones((n, 1))
    X = np.concatenate((ones_col, x_train), axis=1)

    #matrix X^T X
    LHS = np.dot(np.transpose(X), X)

    #matrix X^T Y
    RHS = np.dot(np.transpose(X), y_train)

    #solution beta to normal equations, since LHS is invertible by toy construction
    betas = np.dot(np.linalg.inv(LHS), RHS)
    
    return betas

beta0 = simple_linear_regression_fit(x_train, y_train)[0]
beta1 = simple_linear_regression_fit(x_train, y_train)[1]

print("(beta0, beta1) = (%f, %f)" %(beta0, beta1))


#Ploteo del ajuste


f = lambda x:beta0+beta1*x

#UN arreglo de 0 hasta 4 de aumento de 0.01
xfit=np.arange(0,4,.01)
yfit=f(xfit)

plt.plot(x_train, y_train, 'ko', xfit, yfit)
plt.xlabel('x')
plt.ylabel('y')


#BUILDING A MODEL WITH STATSMODEL AND SKLEARN

#STATSMODEL

#create the X matrix by appending a column of ones to x_train
X = sm.add_constant(x_train)
#this is the same matrix as in our scratch problem!
print(X)
#build the OLS model (ordinary least squares) from the training data
toyregr_sm = sm.OLS(y_train, X)
#save regression info (parameters, etc) in results_sm
results_sm = toyregr_sm.fit()
dir(results_sm)
#pull the beta parameters out from results_sm
beta0_sm = results_sm.params[0]
beta1_sm = results_sm.params[1]

print("(beta0, beta1) = (%f, %f)" %(beta0_sm, beta1_sm))


#SKLEARN

#build the least squares model
toyregr_skl = linear_model.LinearRegression()
#save regression info (parameters, etc) in results_skl
results_skl = toyregr_skl.fit(x_train,y_train)
#pull the beta parameters out from results_skl
beta0_skl = results_skl.intercept_
beta1_skl = results_skl.coef_[0]

print("(beta0, beta1) = (%f, %f)" %(beta0_skl, beta1_skl))

#REGRESIÓN POLINOMIAL

#Para el caso de sklearn, cuand

x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y);
plt.xlabel('x')
plt.ylabel('y')


from sklearn.linear_model import LinearRegression
X=x.reshape(len(x),1)
#O de manera equivalente

X = x[:, np.newaxis]
model = LinearRegression().fit(X, y)
yfit = model.predict(X)
plt.scatter(x, y)
plt.plot(x, yfit);

#Importando un modulo polinomial

from sklearn.preprocessing import PolynomialFeatures
#Creando el objeto polinomial
poly = PolynomialFeatures(degree=3)
#Ajustando los datos al objeto
X2= poly.fit_transform(X)
print(X2)
#En l aprimera columna se tiene los unos del intercepto
#en la segunda se tiene la x normal
#en la tercera la x al cuadrado
#en la cuarta la x al cubo

#Ajustando el modelo polinomial

model= LinearRegression().fit(X2,y)
yfit = model.predict(X2)
plt.scatter(x, y)
plt.plot(x, yfit);


# =============================================================================
# PARTE 2
# =============================================================================

#SIMPLE LINEAR REGRESSION WITH AUTOMOBILE DATA

dfcars = pd.read_csv("/home/user/anaconda3/lml2018/Labs/Lab2/data/mtcars.csv")
dfcars.head()#Note que el primer nombre está horrible
dfcars=dfcars.rename(columns={"Unnamed: 0":"name"})

#Veamos las dimesiones del conjunto de datos
dfcars.shape
#Dividiremos en conjunto de entrenamiento y testeo

from sklearn.model_selection import train_test_split
#set random_state to get the same split every time

traindf, testdf = train_test_split(dfcars, test_size=0.2, random_state=42)


#EXERCISE: Pick one variable to use as a predictor for simple linear 
#regression. Create a markdown cell below and discuss your reasons. 
#You may want to justify this with some visualizations. Is there a second 
#variable you'd like to use as well, say for multiple linear regression 
#with two predictors?

import seaborn as sn
sn.pairplot(dfcars.iloc[:,1:7])
sn.pairplot(dfcars.iloc[:,[1,7,8,9,10,11]])

#De estos gráfico se nota tendencia del Mpg con las variables CYL,DISP,HP,DRAFT,WT




#
#    EXERCISE: With either sklearn or statsmodels, fit the training data using simple linear regression. Use the model to make mpg predictions on testing set.
#
#    Plot the data and the prediction.
#
#    Print out the mean squared error for the training set and the testing set and compare.




#USANDO SOLO HP y WT
x_train=traindf.loc[:,["wt","hp"]]
y_train=traindf.loc[:,["mpg"]]

x_test=testdf.loc[:,["wt","hp"]]
y_test=testdf.loc[:,["mpg"]]

model=LinearRegression().fit(x_train,y_train)

y_hat=model.predict(x_train)
y_hat_test=model.predict(x_test)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_train, y_hat))
print(mean_squared_error(y_test, y_hat_test))


fig,ax =plt.subplots(1,1)
ax.scatter(y_hat,y_train)
ax.set_ylim(10,35)
ax.set_xlim(10,35)







#SOLUCIÖN DEL TEXTO

#your code here
#define  predictor and response for training set
y_train = traindf.mpg
x_train = traindf[['wt']]

# define predictor and response for testing set
y_test = testdf.mpg
x_test = testdf[['wt']]


plt.scatter(dfcars['wt'],dfcars['mpg'], alpha=0.2,color='b',label='test set')
plt.scatter(traindf['wt'],traindf['mpg'],color='b',label='training set')
plt.xlabel('wt')
plt.ylabel('mpg')
plt.legend()


#your code here
# create linear regression object with sklearn
regr = linear_model.LinearRegression()

#your code here
# train the model and make predictions
regr.fit(x_train, y_train)


y_pred = regr.predict(x_test)
#your code here
#print out coefficients
print('Coefficients: \n', regr.coef_[0], regr.intercept_)



# Plot outputs for test set

x0 = np.linspace(2.0,5.5,100)
y0 = regr.coef_[0]*x0 + regr.intercept_

#print(x_test.values)

plt.scatter(x_test.values, y_test.values, color="black",label="Reales")
plt.scatter(x_test.values, y_pred, color="blue",label="Predichos")
plt.plot(x0,y0)
plt.legend()
#plt.scatter(x_test, y_test, color="black")
#plt.scatter(x_train,regr.predict(x_train), color="red")

plt.xlabel('wt')
plt.ylabel('mpg')


# Plot outputs for training set

x0 = np.linspace(1.5,5.5,100)
y0 = regr.coef_[0]*x0 + regr.intercept_
plt.plot(x0,y0)

plt.scatter(x_train.values, y_train, color="black",label="reales")
plt.scatter(x_train.values, regr.predict(x_train), color="blue",label="Predichos")
#plt.scatter(x_train,regr.predict(x_train), color="red")
plt.legend()
plt.xlabel('wt')
plt.ylabel('mpg')

#print(x_train.values)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_train, regr.predict(x_train)))
print(mean_squared_error(y_test, regr.predict(x_test)))



# =============================================================================
# PARTE 3
# =============================================================================
#MULTIPLE LINEAR REGRESSION WITH AUTOMOBILE DATA 




#    EXERCISE: With either sklearn or statsmodels, fit the training data using multiple linear regression with two predictors. Use the model to make mpg predictions on testing set. Print out the mean squared error for the training set and the testing set and compare.
#
#    How do these training and testing mean squared errors compare to those from the simple linear regression?
#
#    Time permitting, repeat the training and testing with three predictors and calculate the mean squared errors. How do these compare to the errors from the one and two predictor models?

#NOTA: ES LA SOLUCIÓN QUE HICE ANTERIORMENTE!!!!!!!!!!


#SOLUCIÓN DEL TEXTO

#your code here
x_train2 = traindf[['wt', 'hp']]
x_test2 = testdf[['wt', 'hp']]

#create linear regression object with sklearn
regr2 = linear_model.LinearRegression()

#train the model 
regr2.fit(x_train2, y_train)

#make predictions using the testing set
y_pred2 = regr2.predict(x_test2)

#coefficients
print('Coefficients: \n', regr.coef_[0], regr.intercept_)

train_MSE2= np.mean((y_train - regr2.predict(x_train2))**2)
test_MSE2= np.mean((y_test - regr2.predict(x_test2))**2)
print("The training MSE is %2f, the testing MSE is %2f" %(train_MSE2, test_MSE2))








#--------------------------------------------------------------------------
# Now with three predictors

x_train3 = traindf[['wt', 'hp', 'disp']]
x_test3 = testdf[['wt', 'hp', 'disp']]

#create linear regression object with sklearn
regr3 = linear_model.LinearRegression()

#train the model 
regr3.fit(x_train3, y_train)

#make predictions using the testing set
y_pred3 = regr3.predict(x_test3)

#coefficients
print('Coefficients: \n', regr3.coef_, regr3.intercept_)

train_MSE2= np.mean((y_train - regr3.predict(x_train3))**2)
test_MSE2= np.mean((y_test - regr3.predict(x_test3))**2)
print("The training MSE is %2f, the testing MSE is %2f" %(train_MSE2, test_MSE2))

print(mean_squared_error(y_train, regr3.predict(x_train3)))
print(mean_squared_error(y_test, regr3.predict(x_test3)))
print(mean_squared_error(y_train, regr2.predict(x_train2)))
print(mean_squared_error(y_test, regr2.predict(x_test2)))



# =============================================================================
# PARTE 4
# =============================================================================

#KNNEIGHBORS

#Importando la parte adecuada del modulo

from sklearn.neighbors import KNeighborsRegressor
knnreg = KNeighborsRegressor(n_neighbors=5)
knnfit=knnreg.fit(x_train,y_train)

#Medida de ajuste para el conjunto de validación
r2=knnreg.score(x_test,y_test)
r2


#Exercise
#What is the R2 score on the training set?

r2_train=knnreg.score(x_train,y_train)

#CALCULANDO MANUALMENTE EL R CUADRADO DEL MODELO

#Promedio de las respuestas
ybarra=np.mean(y_train)

#Predicciones del modelo
y_hat=knnreg.predict(x_train)

#Suma de cuadrados totales
sstotal= np.sum((y_train-ybarra)**2 )

#Suma de cuadrados residuales
ssres=np.sum( (y_train-y_hat)**2 )

R2_manual= 1- ssres/sstotal
R2_manual==r2_train



#Veamos cual sería el K optimo para el problema


regdict= {}

for k in [1,2,4,6,8,10,15]:
    knnreg=KNeighborsRegressor(n_neighbors=k)
    knnreg.fit(x_train,y_train)
    print(knnreg.score(x_train,y_train))# print the R2 score in each case.
    regdict[k] = knnreg # Store the regressors in a dictionary


#Creando el objeto del ploteo
fig,ax = plt.subplots(1,1,figsize=(10,6))

#Gráficando
ax.plot(dfcars.wt,dfcars.mpg,"o",color="blue",label="data")

#Creando una regilla para el eje x
xgrid= np.linspace(np.min(dfcars.wt),np.max(dfcars.wt),100)

#Creando las predicciones para cada Knn

for k in [1,2,6,10,15]:
    predictions =regdict[k].predict(xgrid.reshape(100,1))
    ax.plot(xgrid,predictions,label="{0}-NN".format(k))

ax.legend()


#CAMBIANDO LA DEFINICIÓN DE LA MEDIDA DE DISTANCIA PARA EL ALGORITMO

regdict = {}
# Do a bunch of KNN regressions
for k in [1, 2, 4, 6, 8, 10, 15]:
    knnreg = KNeighborsRegressor(n_neighbors=k,p=3)#Donde p es el numéro de afecta la raiz y la potencia de la suma de distancias
    knnreg.fit(x_train, y_train)
    print(knnreg.score(x_train, y_train)) # print the R2 score in each case.
    regdict[k] = knnreg # Store the regressors in a dictionary



#Creando el gráfico
fig,ax = plt.subplots(1,1,figsize=(10,6))

ax.plot(dfcars.wt,dfcars.mpg,"o",label="data")#donde o indica el tipo de linea???

xgrid=np.linspace(np.min(dfcars.wt),np.max(dfcars.wt),100)

#Creando predicciones

for k in [1,6,15]:
    predictions= regdict[k].predict(xgrid.reshape(100,1))
    ax.plot(xgrid,predictions,label="{}-NN".format(k))

ax.legend()



#AHORA INTENTEMOS UN CAMBIO CON LOS PESOS DE LOS VECINOS

regdict = {}
# Do a bunch of KNN regressions
for k in [1, 2, 4, 6, 8, 10, 15]:
    knnreg = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knnreg.fit(x_train, y_train)
    print(knnreg.score(x_train, y_train)) # print the R2 score in each case.
    regdict[k] = knnreg # Store the regressors in a dictionary


# Now let's plot it all
fig, ax = plt.subplots(1,1, figsize=(10,6))

ax.plot(dfcars.wt, dfcars.mpg, 'o', label="data")

xgrid = np.linspace(np.min(dfcars.wt), np.max(dfcars.wt), 100)
for k in [1, 2, 6, 10, 15]:
    predictions = regdict[k].predict(xgrid.reshape(100,1))
    if k in [1, 6, 15]:
        ax.plot(xgrid, predictions, label="{}-NN".format(k))
    

ax.legend();



#
#EXERCISE: Now do knn regression for the case of two or more predictors.
# For each case, calculate the R2 score. Use different values of k, just
# like above. Try different distance and weigthing strategies.
# Ellaborate about your findings.
#


regdict = {}
# Do a bunch of KNN regressions
for k in [1, 2, 4, 6, 8, 10, 15]:
    knnreg = KNeighborsRegressor(n_neighbors=k)
    knnreg.fit(x_train2, y_train)
    print(knnreg.score(x_train2, y_train)) # print the R2 score in each case.
    regdict[k] = knnreg # Store the regressors in a dictionary
    
    
# =============================================================================
# BOOTSRAP
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('seaborn-darkgrid')

# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma


#GRAFICA
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,4))
axes[0].scatter(X1, Y)
axes[1].scatter(X2, Y)
axes[0].set_ylabel('Y'); 
axes[0].set_xlabel('X1'); 
axes[1].set_xlabel('X2');



slopes = np.random.normal(loc=1.0,scale=0.3,size=300)
intercepts = np.random.normal(loc=1,scale=0.3,size=300)


plt.scatter(X1, Y)
x0 = np.linspace(-3.0,2.5,100)
for i in range(len(slopes)):
    plt.plot(x0,x0*slopes[i]+intercepts[i],color='b',alpha=0.1)
plt.ylabel('Y'); plt.xlabel('X1')



#Distribución de las pendientes
fig,ax =plt.subplots(1,1)
ax.hist(slopes)
ax.set_xlabel('Slope')


# Bootstrapping
M_samples=10000  # The number of bootstrap samples we want
N_points = 100  # The number of points we want to samples from the dist.

# Let's sample with replacement.
bs_np = np.random.choice(slopes, size=(M_samples, N_points), replace=True)

# Calculate the mean
sd_mean=np.mean(bs_np, axis=1)#Promedio sobre cada una de las 10000 simulaciones

# And the standard deviation
sd_std=np.std(bs_np, axis=1)

# Plot results
plt.hist(sd_mean, bins=30, normed=True, alpha=0.5,label="samples");
plt.axvline(slopes.mean(), 0, 1, color='r', label='Our Sample')
plt.legend()








