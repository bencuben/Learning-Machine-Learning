#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 13:03:39 2020

@author: Brahian Cano Urrego
"""


# =============================================================================
# LAB 1 
# =============================================================================


# =============================================================================
# PARTE 1
# =============================================================================
#importando los modulos
import numpy as np

#Medias

np.mean([2,5,6,8,9])

#Diferentes tipos de división

#division normal
1/2

#division entera (entero mas cercano al numero)

3//2


#EJEMPLO DE PRINT

print(1+4,"\n","salto")
5/3


#TIPOS DE OBJETOS

#enteros
a=2
#flotantes
b=0.66
#cadena
c="Hola mundo"
#boleano
d=True
#lista esta puede contener multiples tipos de datos
e=[1,2,3,"a"]


#Nota: una lista se declara con () parentesis y una lista con  [] corchetes
a = 1
b = 2.0
a + a, a - b, b * b, 10*a #el resultado es una cuadropla

#COMPARATIVOS

type(a)==int#verdad
type(a)==float#falso
type(a)==str#falso
type(a)==list#falso



#EJERCICIO 1



#    EXERCISE: Create a tuple called tup with the following seven objects:
#
#        The first element is an integer of your choice
#        The second element is a float of your choice
#        The third element is the sum of the first two elements
#        The fourth element is the difference of the first two elements
#        The fifth element is first element divided by the second element
#
#Display the output of tup. What is the type of the variable tup? What happens 
#if you try and change an item in the tuple?

a=1
b=2.0
tup=(a,b,a+b,a-b,a/b)
tup
type(tup)
#tup[0]=111#ERROR Las tuplas no son modificables



#longitud de una lista o tupla
len(tup)

# =============================================================================
# PARTE 2
# =============================================================================

#INDEXAR UNA LISTA


#Python en 0 indexado
tup[0]

#Acceder al último
tup[-1]

#acceder al penultimo
tup[-2]

#definir listas

empty_list = []
float_list = [1., 3., 5., 4., 2.]
int_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mixed_list = [1, 2., 3, 4., 5]
print(empty_list)
print(int_list)
print(mixed_list, float_list)

#Recorrer desde el primero hasta antes del tercero
int_list[:2]

#Recorrer hasta antes del último
int_list[:-1]

#Recorrer todo
int_list[:]

#Recorrer todo con ciertos saltos

int_list[::2]



#ITERANDO UNA LISTA

#TOmando cada elemento
for elem in float_list:
    print(elem)
    
#accediendo por indice
    
for i in range(len(mixed_list)):
    print(mixed_list[i])

#NOTA: Funcion importante enumerate
#esta función devuelve pares de indice mas elemento
for i,elem in enumerate(float_list):
    print(i,elem)

#Para poder ver el objeto enumerate
list(enumerate(float_list))

#Para poder ver el objeto range
list(range(1,10,2))



#Añadiendo elemento a una lista

#manera 1: se le concatena otra lista con +

float_list+[3.33]
len(float_list)
float_list
#No se guardan los cambios

#Manera 2: Append metodo

float_list.append(.33)
len(float_list)
#Afecta a la lista


#Para remover un elemento se usa
#acorde al indice
float_list.pop(2)

del(float_list[2])

#Borra todos los elementos de la lista
float_list.clear()


empty_list = []
float_list = [1., 3., 5., 4., 2.]
int_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mixed_list = [1, 2., 3, 4., 5]


#COMPRESION DE LISTAS

squaredlist = [i*i for i in range(1,11)]
squaredlist

#ESta estructura admite condicionales

comp_list1 = [i**2 for i in range(1,11) if i%2==0]
comp_list1


#EJERCICIO 2


#    EXERCISE: Build a list that contains every prime number between 1 and 100, in two different ways:
#
#       Using for loops and conditional if statements.
#      (Stretch Goal) Using a list comprehension. You should be able to do this in one line of code, and it may be helpful to look up the function all in the documentation.


#Definir función para identificar si un solo numero es primo
def  primo(num):
    return([False for elem in range(2,num) if num%elem ==0])#Los primos arrojan listas vacias

#Iterar la función a través del rango deseado 
[i for i in range(3,101) if len(primo(i))==0]


#Solución 2:

primos = [2 ,	3 	,5 ,	7 ,	11, 13, 	17 ,	19, 	23 ,	29 ,	31 ,	37 ,	41 ,	43 ,	47 ,	53 ,	59 ,	61 	,67 	,71 ,	73,79 ,	83 ,	89 	,97]
print(primos)

primos2=[]
for i in range(0,99):
    if i in primos:#verifica la pertenencia de un elemento a una lista
        primos2.append(i)
print(primos2)


# =============================================================================
# PARTE 3
# =============================================================================
#STRINGS AND LISTINESS

#esto es listable
astring="kevin"

#tiene longitud
len(astring)

#se puede dividir

astring[0:3]

# Y se pude iterar

for i in astring:
    print(i)

#otra manera

i=0
while i<len(astring):
    print(astring[i])
    i=i+1

#Concatecar strings
    
astring+" es una"+" valija"



#EXERCISE




#    EXERCISE: Make three strings, called first, middle, and last, with your first, middle, and last names, respectively. If you don't have a middle name, make up a middle name!
#
#   Then create a string called full_name that joins your first, middle, and last name, with a space separating your first, middle, and last names.
#
#    Finally make a string called full_name_rev which takes full_name and reverses the letters. For example, if full_name is Jane Beth Doe, then full_name_rev is eoD hteB enaJ.

first="Brahian"
middle="Cano"
last="Urrego"

full_name=first+" "+middle+ " "+last
full_name


full_name_rev=full_name[-1:-(len(full_name)+1):-1]
full_name_rev



# =============================================================================
# PARTE 4
# =============================================================================
#DICCIONARIOS


#Definicion de un diccionario se hacer con {}
enroll2016_dict = {'CS50': 692, 'CS109 / Stat 121 / AC 209': 312, 'Econ1011a': 95, 'AM21a': 153, 'Stat110': 485}
enroll2016_dict

#Se accede por la clave del "objeto"
enroll2016_dict['CS50']

#Los valores de los objetos
enroll2016_dict.values()

#Devuelve las tuplas del diccionario
enroll2016_dict.items()


for key,value in enroll2016_dict.items():
    print("%s:%d" %(key,value))
    
    
second_dict={}
for key in enroll2016_dict:
    second_dict[key] = enroll2016_dict[key]
second_dict

#COmpresión de diccionarios

my_dict={k:v for (k,v) in zip(int_list, float_list)}
my_dict

#Donde Zip genera tuplas con los elemento del mismo indice en ambas listas
list(zip(int_list, float_list) )


#Otra manera para generar un diccionario es con la función dict
dict(a=1,b=2)


#NOTA SOBRE LOS ITERADORES:

#Tenes sons exhaustivos lo que quiere decir que se consumen
#ej

iterador= enumerate(astring)

#Tipo enumerate
type(iterador)

#En la primera se consumiran los elementos
for i,elem in iterador:
    print(i,elem)
    
#POr lo que en esta no habran elementos    
for i,elem in iterador:
    print(i,elem)
  
# =============================================================================
# Parte 5
# =============================================================================

#FUNCIONES
    
print(float_list)
#Donde append es una metodo de la clase lista
float_list.append(56.7) 
float_list


#Lambda functions

#Often we define a mathematical function with a quick one-line function called a lambda. No return statement is needed.
#
#The big use of lambda functions in data science is for mathematical functions.

#Definición en una linea de la función cuadrado
square = lambda x: x**2
print(square(3) )

#Definición de multiplicación
multiplicacion = lambda x,y: x*y
print(multiplicacion(3,4))

#EXERCISE: 
#Write a function called isprime that takes in a positive integer N, and determines whether or not it is prime. Return the N
#
#if it's prime and return nothing if it isn't. You may want to reuse part of your code from the exercise in Part 2.
#
#Then, using a list comprehension and isprime, create a list myprimes that contains all the prime numbers less than 100.

#Pone marcadores para saber si un numero es primero o no
def isprime(n):
    if n in primos:
        return("1")
    else:
        return("0")

[i for i in range(1,101) if isprime(i)=="1"]


#argumentos por defecto

def get_multiple(x,y=1):
    return x*y

print("With x and y:", get_multiple(10, 2))
print("With x only:", get_multiple(10))


#Otro ejemplo
def print_special_greeting(name, leaving = False, condition = "nice"):
    print("Hi", name)
    print("How are you doing on this", condition, "day?")
    if leaving:
        print("Please come back! ")

# Use all the default arguments.
print_special_greeting("Pavlos")

#Changing them
print_special_greeting("Pavlos", True, "rainy")


#Positional an keywords arguments


#POSITIIONAL
#Todo lo que sigue despues del primer argumento quedará en una tupla
#llamada siblings.
def print_siblings(name,*siblings):
    print(name,"has the following siblings:")
    for sib in siblings:
        print(sib)
    print(type(siblings))
    
print_siblings("John", "Ashley", "Lauren", "Arthur")
print_siblings("Mike", "John")
print_siblings("Terry")


#KWARGS
#Con doble asterisco Los argumentos que se pasen opcional quedarán en un diccionario
def print_brothers_sisters(name, **siblings):
    print(name, "has the following siblings:")
    for sib in siblings:
        print(sib, ":", siblings[sib])
    print(type(siblings))
    
print_brothers_sisters("John", Ashley="sister", Lauren="sister", Arthur="brother")        

#Nota:

#Finally, when putting all those things together one must follow a certain order:
# Below is a more general function definition. The ordering of the inputs is key:
# arguments, default, positional, keyword arguments.


#EJEMPLO:

def f(a, b, c=5, *tupleargs, **dictargs):
    print("got", a, b, c, tupleargs, dictargs)
    return a
print(f(1,3))
print(f(1, 3, c=4, d=1, e=3))
print(f(1, 3, 9, 11, d=1, e=3)) # try calling with c = 9 to see what happens!


#SE PUEDE USAR FUNCIONES DENTRO DE FUNCIONES

def sum_of_anything(x, y, f):
    print(x, y, f)
    return(f(x) + f(y))
    
sum_of_anything(3,4,square)
sum_of_anything(3,4,isprime)



#EXERCISE: Create a dictionary, called ps_dict, that contains with 
#the primes less than 100 and their corresponding squares.

# your code here

#En diccionario comprimido
pd_dict = {i:i**2 for i in primos}

#Solucion larga
print(primos)
pd_dict = {}
for i in primos:
    pd_dict[i] = i**2

pd_dict



# =============================================================================
# PARTE 6
# =============================================================================

#EXEPCION HANDLING

def bad_func(x:float, y:float) -> float:
    try:
        result=x/y
    except ZeroDivisionError:
        print("WARNING:")
        print("OMG you set y=0 ")
        print("We are setting y = 1.  This may drastically change your results.")
        y = 1.0
        result = x/y
    return(result)

x, y = 1.0, 0.0
important_quantity = bad_func(x, y)
print("\n Your important_quantity has a value of {0:3.6f}".format(important_quantity))

# =============================================================================
# PARTE 7
# =============================================================================

#NUMPY

#Definiendo una lista como un arreglo numpy
my_array = np.array([5,10,15,20])
my_array
print(my_array)

#LOS OBJETOS NUMPY SON LISTABLES

#Usando los metodos de esta clase

#primera forma
print(my_array.mean())
#Segunda forma
print(np.mean(my_array))

#Muestas aleatoria de una normal 0, 1
normal_array = np.random.randn(1000)
print("The sample mean and standard devation are %f and %f, respectively." %(np.mean(normal_array), np.std(normal_array)))

#Numpy usa el concepto de broadcasting
#Que permite modificar cada elemento con operaciones basicas tal com

normal_5_7 = 5 + 7*normal_array
np.mean(normal_5_7), np.std(normal_5_7)

#ALGUNOS CONSTRUCTORES DE ARREGLOS

zeros = np.zeros(10) # generates 10 floating point zeros
zeros

ones = np.ones(3)
ones

#NOTA: todos los elemento de un arreglo deben ser del mismo tipo

zeros.dtype

np.ones(10, dtype='int') # generates 10 integer ones

#Numeros aleatorios de un uniforme [0,1]
np.random.rand(10)

#OPERACIONES DE VECTORES
ones_array = np.ones(5)
twos_array = 2*np.ones(5)
ones_array + twos_array

#SI FUERA CON LISTAS FUERA DIFERENTE

first_list = [1., 1., 1., 1., 1.]
second_list = [2., 2., 2., 2., 2.]
first_list + second_list # not what you want


#ARREGLOS 2-D

#Usando listas de listas

my_array2d=np.array(
        [[1,0,0],
         [0,1,0],
         [0,0,1]
                ]
        )
print(my_array2d)


# you can do the same without the pretty formatting (decide which style you like better)
my_array2d = np.array([ [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12] ])

# 3 x 4 array of ones
ones_2d = np.ones([3, 4])
print(ones_2d, "\n")

# 3 x 4 array of ones with random noise
ones_noise = ones_2d + 0.01*np.random.randn(3, 4)
print(ones_noise, "\n")

# 3 x 3 identity matrix
my_identity = np.eye(3)
print(my_identity, "\n")

#NOTA: estos arreglados son 0 indexados

print(my_array2d)
print("element [2,3] is:",my_array2d[1,2])

#Se puede iterar ademas de dividirlo

my_array2d[1:,3]

#NOTA:

#The axis 0 is the one going downwards (i.e. the rows), whereas axis 1 
#is the one going across (the columns). You will often use functions such as 
#mean or sum along a particular axis. If you sum along axis 0 you are summing 
#across the rows and will end up with one value per column. As a rule, any axis
# you list in the axis argument will dissapear.


np.sum(ones_2d, axis=0)
np.sum(ones_2d, axis=1)


#Exercise
#* Create a two-dimensional array of size 3×5
#and do the following: * Print out the array * Print out the shape of the array * 
#Create two slices of the array: 1. The first slice should be the last row and the 
#third through last column 2. The second slice should be rows 1−3 and columns 3−5 * 
#Square each element in the array and print the result


A = np.array([ [5, 4, 3, 2, 1], [1, 2, 3, 4, 5], [1.1, 2.2, 3.3, 4.4, 5.5] ])
print(A, "\n")

A.shape

#slicing

A[-1,2:]

A[0:2,2:4]

A*A

#Matrix operations

three_by_four = np.ones([3,4])
three_by_four

#TRASPONER UN ARREGLO
four_by_three = three_by_four.T

#Matrix multiplication is accomplished by np.dot. The * operator will do element-wise multiplication

print(np.dot(three_by_four, four_by_three)) # 3 x 3 matrix
np.dot(four_by_three, three_by_four) # 4 x 4 matrix

matrix = np.random.rand(4,4) # a 4 by 4 matrix
matrix

#EIGEN VALORES DE UNA MATRIZ
np.linalg.eig(matrix)

#INVERSA
inv_matrix = np.linalg.inv(matrix) # the invert matrix
print(inv_matrix)

#prove it's the inverse
np.dot(matrix,inv_matrix)

#Resuelte un sistema básico de ecuaciones
#np.linalg.solve?

# =============================================================================
# PARTE 8
# =============================================================================

#Procesamiento de lenguaje natural

#Abrir un archivo de texto

# Open the file for reading
f = open("/home/user/anaconda3/lml2018/Labs/Lab1and3/data/brief_comments.txt","r")

#reading the file
dogs = f.read()
#Closing the file
f.close()


# This approach is the correct way, and should always be used.
with open("/home/user/anaconda3/lml2018/Labs/Lab1and3/data/brief_comments.txt", "r") as f:
    dogs = f.read()
    
#Preprocesamiento

#El contenido del texto
print(dogs)
#El tipo de objeto
type(dogs)
#La longitud de la cadena
l = len(dogs) # How many characters are in this string?
print(l)

#Dividamos el texto en palabras

words = dogs.split()#Almacena cada palabra en una lista
words


type(words)


N = len(words) # Number of words
print("There are {0} words in our brief comments.".format(N))

#Cuantas veces aparece el elemento "dogs" en la lista??
words.count("dogs")
dogs

#Tenemos que limpiar el texto para acceder a las palabras
#Se quitan todos los puntos y se pone cada palabra en minuscula
more_words = [word.split(".")[0].lower() for word in words]


more_words.count("dogs")

# Your code here
#opening data
with open("/home/user/anaconda3/lml2018/Labs/Lab1and3/data/hamlet.txt","r") as f:
    hamlettext = f.read()

#Type,len
print(type(hamlettext))
print(len(hamlettext),"\n")
print(hamlettext[:500])

#creation of list
hamletwords = hamlettext.split()
hamletwords1 = [i.split(".")[0] for i in hamletwords]
hamletwords2 = [i.split(",")[0] for i in hamletwords1]
hamletwords3 = [i.split(";")[0] for i in hamletwords2]

print(hamletwords2[:10])
N =len(hamletwords)
print("there are {0} words in Hamlet".format(N))

#lower case list
hamletwords_lc = [i.lower() for i in hamletwords3]

print(hamletwords_lc.count("thou"))

#la función set crea una lista con los elementos unicos
a = len(set(hamletwords_lc))
print("there are {0} words in Hamlet".format(a))


#Writing files

#Creemos vector para guardar
my_ints = [i for i in range(-5,6)]

my_ints2 = [i**2 for i in my_ints]

print("Our list is {0}".format(my_ints2))

#Forma complicada

with open("/home/user/anaconda3/lml2018/Labs/Lab1and3/data/datafile.txt","w") as dataf:
    #header
    dataf.write("Here is a list of squared ints. \n \n")
    #Colums
    dataf.write("n")
    dataf.write(", ")
    dataf.write("n^2"+"\n")
    #data
    for i,i2 in zip(my_ints,my_ints2):
        dataf.write("{},{}\n".format(str(i),str(i2)))
        
        
        
        
#Usando Json
        
import json

dog_shelter = {} # Initialize dictionary

# Set up dictionary elements
dog_shelter['dog1'] = {'name': 'Cloe', 'age': 3, 'breed': 'Border Collie', 'playgroup': 'Yes'}
dog_shelter['dog2'] = {'name': 'Karl', 'age': 7, 'breed': 'Beagle', 'playgroup': 'Yes'}

dog_shelter

#Devolviendo los diccionarios de cada perro
dog_shelter.values()

#veamos los valores del perro1
dog_shelter["dog1"]

dog_shelter["dog2"]["name"]

#Guardar este diccionario en formato Json

with open("/home/user/anaconda3/lml2018/Labs/Lab1and3/data/dog_shelter_info.txt","w") as output:
    json.dump(dog_shelter,output)

#Leyendo el archivo Json
    
with open("/home/user/anaconda3/lml2018/Labs/Lab1and3/data/dog_shelter_info.txt","r") as f:
    dog_data=json.load(f)

print(dog_data)



#Exploremos el diccionario

for dogid, info in dog_data.items():
    print(dogid)
    print("{0} is a {1} year old {2}.".format(info['name'], info['age'], info['breed']))
    if info['playgroup'].lower() == 'yes':
        print("{0} can attend playgroup.".format(info['name']))
    else:
        print("{0} is not permitted at playgroup.".format(info['name']))
    print("======================================\n")
    
# =============================================================================
# PARTE 9
# =============================================================================

#Exoresiones REGULARES
    
birthday="June 11"

#Extraer el mes de la forma más básica
birthday.strip()[:-3]

#Con expresion regular
regex=r"\w"#a first regular expression

#The r means that the string is a raw string. 
#This just tells python not to interpret backslashes and other metacharacters 
#in the string.

#The \w indicates any alphanumeric character.
#The + indicates one or more occurances.

    
type(regex)

import re#Regular expression module
month=re.search(regex,birthday)
print(month)
#El argumento span( x,y)
#Denota x=comienzo del caracter buscado
#y=Final del caracter buscado


#otro ejemplo
regex = r"June"
re.search(regex, birthday)


#CUando falla un reges
re.search(r"Oct", birthday) # nothing prints out


#Exercise
#Consider the string ```python statement = "June is a lovely month." ``` *
# Use a regular expression to the find the pattern `June`.
# * Create a new string, `fragment` from `statement`, which starts just
# after the word `June`.

#Your output should be is a lovely month.



statement = "June is a lovely month."
regex=r"June"
junio=re.search(regex,statement)
fragment=statement[(junio.end()+1):]

#Otro tipo de comando para regex

regex = r"\d+"#Busca ocurrenncia de DIGITOS
re.search(regex,birthday)
#El comando + es para que encuentre mas de una ocurrencia

cadena= "Hola gente 10"
regex= r"\w+"
re.search(regex,cadena)
#Con esto se demuestra que devuelve toda una frase hasta un espacio


cadena2= "Hola10    "
regex= r"\w+"
re.search(regex,cadena2)#No toma en cuenta espacios

#Si queremos solos el número

regex= r"\d+"
re.search(regex,cadena2)

#Si queremos solo el texto

#Letras En MAYUS o en minus
regex=r"[A-Za-z]+"
re.search(regex,cadena2)

#Otra manera de obtener el número es
regex=r"[0-9]+"
re.search(regex,cadena2)


#FINDALL()
#Con este se devuelve listas con palabras que cumplan con el regex


regex_texto=r"[a-zA-Z]+"
re.findall(regex_texto,cadena)


#Busquemos numeros
regex_day = r"\d+"
re.findall(regex_day, cadena)



#GROUPS
birthdays = "June 11th, December 13th, September 21st, May 12th"

#ej:
regex = r"([A-Za-z]+) (\d+\w+)"#
#Nota:(\d+\w+): Digitos seguidos de alfanumericos
bdays = re.findall(regex, birthdays)
print(bdays)

#OTRAS OPCIONES PARA LO MISMO
regex = r"([A-Za-z]+)\s(\d+\w+)"
regex = r"([A-Za-z]+)\s(\w+)"
regex = r"([A-Za-z]+) (\d+[a-z]+)"

#NOte que el espacio se puede representar con \s


#Tambien se puede obtener solo el mes y el dia solo

regex = r"[A-Za-z]+ \d+"
bdays = re.findall(regex, birthdays)
for bday in bdays:
    print(bday)

#O en versión de grupos
    
regex = r"([A-Za-z]+) (\d+)"
re.findall(regex, birthdays)

#Prueba

cadena= "Junio 11, Mayo error 15" 
regex = r"([A-Za-z]+) (\d+)"
re.findall(regex, cadena)

#Busca la congruencia exacta y luego lo separa en las tuplas



with open("/home/user/anaconda3/lml2018/Labs/Lab1and3/data/shelterdogs.xml","r") as f:
    shelterdogs=f.read()

#Exercise
#* Open and read the file `shelterdogs.xml` into a string named `dogs`. It should look like: 
#Write a regular expression to match the dog names. That is, you want to match the name inside the name tag: <name> dog_name </name>
regex= r"<name>\s+(\w+)"
nperros = re.findall(regex, shelterdogs)

for name in nperros:
    print(name)



# =============================================================================
# PART 10
# =============================================================================

#PANDAS
    
import pandas as pd

#Lectura de datos
dfcars=pd.read_csv("/home/user/anaconda3/lml2018/Labs/Lab1and3/data/mtcars.csv")

#Mostrar los primero registros de la base

dfcars.head(10)

#Mostrar los ultimos registros
dfcars.tail(10)

#Nota: las columnas en pandas son llamadas series

#Descriptivo de los datos

dfcars.describe(include="all")#All incluye las var categoricas en el resumen

#Renombrando columnas se hace uso de un diccionaro


dfcars=dfcars.rename(columns={"Unnamed: 0":"car name"})

#Dimensiones del data frame

#a las que no tiene el () se le llama propiedades
dfcars.shape

#Nota: len solo devuelve el numero de filas del data frame
len(dfcars)

#Recorrer los nombres de las columnas
for ele in dfcars:
    print(ele)
    
#es equivalente a la siguiente propiedad
dfcars.columns

#recorrer elemento de una serie
for ele in dfcars.hp:
    print(ele)
    
    
#Index para el data frame
dfcars.index#Info en rango
list(dfcars.index)#Lista con todo el rango

#Tambien las series tienen indices

dfcars.cyl.index


#INDEXAR EN DATA FRAM

# the loc property indexes by label name
# the iloc indexes by position in the index.


#Ejemplo

# create values from 5 to 36
new_index = [i+5 for i in range(32)]

# new dataframe with indexed rows from 5 to 36
dfcars_reindex = dfcars.reindex(new_index)
dfcars_reindex.head()

#ahora con las filas recodificadas

#Donde este busca por indice
dfcars_reindex.iloc[0:3]

#Donde este busca por el alias de la fila
dfcars_reindex.loc[5:7] # or dfcars_reindex.loc[0:7]


#Buscar filas y columnas

dfcars_reindex.iloc[2:5,1:4]
#Note que los nombres de las filas no corresponden al indice real

dfcars_reindex.loc[7:9,["mpg","cyl","disp"]]

#combinacion??
dfcars_reindex.loc[:,["mpg","cyl","disp"]].iloc[0:10]




s1=[0,1,2,3]
s2=[4,5,6,7]
df={"column_1":s1,"column_2":s2}

#Las data frames se crean como diccionaros

table = pd.DataFrame(data=df)
table

#Renombrando columnas

table = table.rename({"column_1":"Col_1","column_2":"Col_2"},axis="columns")
table

nombres= ["cero","uno","dos","tres"]
indice= {i:j for i,j in zip(list(range(0,4)),nombres)}
table = table.rename(indice,axis="index")

#JSON INTO PANDAS
#Se carga el archivo y que almacenado como diccionarios
with open('/home/user/anaconda3/lml2018/Labs/Lab1and3/data/dog_shelter_info.txt', 'r') as f:
    dog_data = json.load(f)

#Se convierte en un string
dog_data_json_str = json.dumps(dog_data)

type(dog_data_json_str)
#esto con el fin de convertirlo en un dataframe
df = pd.read_json(dog_data_json_str)

df

# =============================================================================
# Parte 11
# =============================================================================

#BEATIFUL SOUP


#Tomando datos de la web
import requests as rq

req = rq.get("https://en.wikipedia.org/wiki/Harvard_University")
# This is equivalent to typing a URL into your browser and hitting enter.

req
#Donde Response [200] signfica conexion exitosa

type(req)

#NOTA: para listar las propiedades de un objeto

dir(req)

#Como este objeto solo es una solitud,
#extraigamos el texto de l página

page =req.text
page[20000:30000]

#Como todas estas etiquetas de html son un dolor de cabeza
#es donde entra nuestro aigo

from bs4 import BeautifulSoup
#ahora se convirtio en un objeto especial con propiedades interesantes
soup = BeautifulSoup(page,"html.parser")

#Para printar de manera bonita se una
soup.prettify()[:1000]


#Por otra parte con este nuevo objeto se pueden acceder a las etiquetas HTML

soup.title
soup.h1
#Estas propiedades solo son buena con objeto que aparezcan una sola vez

#Es decir se debe ser cuidadoso con elementos que aparecen multiples veces

#Cada multiles coincidencia se usa:
len(soup.find_all("p"))

#Vamos a buscar las tablas de la web
len(soup.find_all("table") )#26 tablas en total

#Miremos las clases de esas tablas
soup.table["class"]
#
#Busquemos las tablas que tengan alguna clase
[t["class"] for t in soup.find_all("table") if t.get("class")]


#Busca en el tag table con el atributo wikitable
table_demographics = soup.find_all("table", "wikitable")[2]

from IPython.core.display import HTML

HTML(str(table_demographics))
#SOLO FUNCIONA EN JUPYTER

#Para extraer cada fila se usa el argumento "tr"
rows = [row for row in table_demographics.find_all("tr")]
print(rows)

#Encabezado
header_row = rows[0]
HTML(str(header_row))

#convirtiendo todos los saltos de linea a espacios


# Lambda expressions return the value of the expression inside it.
# In this case, it will return a string with new line characters replaced by spaces.
rem_nl = lambda s: s.replace("\n", " ")

# the if col.get_text() takes care of no-text in the upper left
columns = [rem_nl(col.get_text()) for col in header_row.find_all("th") if col.get_text()]
columns

#find solo devuelve la primera ocurrencia 
indexes = [row.find("th").get_text() for row in rows[1:]]
indexes

#Para convertir el valor de estas etiquetas en numero se debe usar
#la siguiente lógica

#Now we want to transform the string on the cells to integers. To do this, we follow a very common python pattern:
#
#    Check if the last character of the string is a percent sign
#    If it is, then convert the characters before the percent sign to integers
#    If one of the prior checks fails, return a value of None

def to_num(s):
    if s[-1]=="%":
        return(int(s[:-1]))
    else:
        return None

values = [to_num(value.get_text()) for row in rows[1:] for value in row.find_all("td")]
values

#ahora el problema es que se perdieron las agrupaciones de los valores 

#Arreglando eso

stacked_values_lists = [values[i::3]  for i in range(len(columns))]
stacked_values_lists

#ahora las volvemos tuplas
stacked_values = zip(*stacked_values_lists)

#Notice the use of the * in front: that converts the list of lists to a set of arguments to zip. See the ASIDE below.

list(stacked_values)


#Resultado
# Here's the original HTML table for visual understanding
HTML(str(table_demographics))


# =============================================================================
# Parte 12
# =============================================================================

#MATPLOTLIB

# Your code here
import numpy as np

def logistic(z: np.ndarray, a: float, b: float) -> np.ndarray:
    """ Compute logistic function
      Inputs:
         a: exponential parameter
         b: exponential prefactor
         z: numpy array; domain
      Outputs:
         f: numpy array of floats, logistic function
    """
    
    den = 1.0 + b * np.exp(-a * z)
    return 1.0 / den

def stretch_tanh(z: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """ Compute stretched hyperbolic tangent
      Inputs:
         a: horizontal stretch parameter (a>1 implies a horizontal squish)
         b: vertical stretch parameter
         c: vertical shift parameter
         z: numpy array; domain
      Outputs:
         g: numpy array of floats, stretched tanh
    """
    return b * np.tanh(a * z) + c

def relu(z: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """ Compute rectificed linear unit
      Inputs:
         eps: small positive parameter
         z: numpy array; domain
      Outputs:
         h: numpy array; relu
    """
    return np.fmax(z, eps * z)

#Nota: estas funciones estan vectorizadas

#Creando los parametros para la grafica
    
x = np.linspace(-5.0, 5.0, 100) # Equally spaced grid of 100 pts between -5 and 5

f = logistic(x, -1.0, 1.0) # Generate data

import matplotlib.pyplot as plt

plt.plot(x,f); # Use the semicolon to suppress some iPython output (not needed in real Python scripts)


#Demole un poco de nombres a los ejes

plt.plot(x, f)
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.title('Logistic Function (main menu)');

#Agregando rejilla
plt.grid()


#Before proceeding any further, I'm going to change notation. The plotting 
#interface we've been working with so far is okay, but not as flexible as it 
#can be. In fact, I don't usually generate my plots with this interface. I work 
#with slightly lower-level methods, which I will introduce to you now. 
#The reason I need to make a big deal about this is because the lower-level
#methods have a slightly different API. This will become apparent in my next example.

#Compilando graficas a nivel mas bajo
#Get figure an axes objetcs
fig, ax = plt.subplots(1,1)

#Se plotea encima de los ejes vacios
ax.plot(x,f);

#Etiquetas
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("logistic function")

ax.grid()

#Objetivos


#    Make labels bigger!
#    Make line fatter
#    Make tick mark labels bigger
#    Make the grid less pronounced
#    Make figure bigger

#figsize para modificar las dimensiones de la figura
fig, ax = plt.subplots(1,1,figsize=(10,6))


ax.plot(x,f,lw=4)#Linea mas gruesa
#Font bigger
ax.set_xlabel("x",fontsize=24)
ax.set_ylabel("f",fontsize=24)
ax.set_title("logistic function",fontsize=24)

#Alterando la rejilla
#alpha indica como la transparencia de la rejilla
#ls=linestyle
ax.grid(True,lw=1.5,ls="--",alpha=0.75)



#Mejorando las marcas de los ejes

fig, ax = plt.subplots(1,1, figsize=(10,6)) # Make figure bigger

# Make line plot
ax.plot(x, f, lw=4)

# Update ticklabel size
ax.tick_params(labelsize=24)

# Make labels
ax.set_xlabel(r'$\theta=x$', fontsize=24) # Use TeX for mathematical rendering
ax.set_ylabel(r'$f(\theta)$', fontsize=24) # Use TeX for mathematical rendering
ax.set_title('Logistic Function', fontsize=24)

ax.grid(True, lw=1.5, ls='--', alpha=0.75)



#POR ULTIMO CAMBIAREMOS LOS LIMITES DE LA GRAFICA

fig, ax = plt.subplots(1,1, figsize=(10,6)) # Make figure bigger

# Make line plot
ax.plot(x, f, lw=4)

ax.set_xlim(x.min(),x.max())

# Update ticklabel size
ax.tick_params(labelsize=24)

# Make labels
ax.set_xlabel(r'$x$', fontsize=24) # Use TeX for mathematical rendering
ax.set_ylabel(r'$f(x)$', fontsize=24) # Use TeX for mathematical rendering
ax.set_title('Logistic Function', fontsize=24)

ax.grid(True, lw=1.5, ls='--', alpha=0.75)



#Exercise
#Do the following:
#
#    Make a figure with the logistic function, hyperbolic tangent, and rectified linear unit.
#    Use different line styles for each plot
#    Put a legend on your figure
#
#Here's an example of a figure: 



# First get the data
f = logistic(x, -2.0, 1.0)
g = stretch_tanh(x, 2.0, 0.5, 0.5)
h = relu(x)

fig, ax = plt.subplots(1,1, figsize=(10,6)) # Create figure object

# Make actual plots
# (Notice the label argument!)
ax.plot(x, f, lw=4, ls='-', label=r'$L(x;1)$')
ax.plot(x, g, lw=4, ls='--', label=r'$\tanh(2x)$')
ax.plot(x, h, lw=4, ls='-.', label=r'$relu(x; 0.01)$')

# Make the tick labels readable
ax.tick_params(labelsize=24)

# Set axes limits to make the scale nice
ax.set_xlim(x.min(), x.max())
ax.set_ylim(h.min(), 1.1)

# Make readable labels
ax.set_xlabel(r'$x$', fontsize=24)
ax.set_ylabel(r'$h(x)$', fontsize=24)
ax.set_title('Activation Functions', fontsize=24)


# Set up grid
ax.grid(True, lw=1.75, ls='--', alpha=0.75)

# Put legend on figure
ax.legend(loc='best', fontsize=24);


#EXERCISE

#* Read the *matplotlib rcParams* section at the following page: [Customizing matplotlib](https://matplotlib.org/users/customizing.html) * Create your very own `config.py` file. It should have the following structure: ```python pars = {} ``` You must fill in the `pars` dictionary yourself. All the possible parameters can be found at the link above. For example, if you want to set a default line width of `4`, then you would have ```python pars = {'lines.linewidth': 4} ``` in your `config.py` file. * Make sure your `config.py` file is in the same directory as your lab notebook. * Make a plot (similar to the one I made above) using your `config` file.

#Usar plantillas predefinidas
#NOTA: Para ver las plantillas disponibles:
print(plt.style.available)
######################

plt.style.use("dark_background")
fig, ax = plt.subplots(1,1, figsize=(10,6)) # Create figure object
ax.plot(x,f,label="line 1")
ax.plot(x,g,label="line 2")
ax.plot(x,h,label="line 3")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title("Probando ggplot")
ax.grid(True,lw=1.1,ls="--",alpha=0.9)
ax.legend(loc="best")

