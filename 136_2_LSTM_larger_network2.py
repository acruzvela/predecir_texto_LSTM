import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import sys
import os

os.system('cls' if os.name=='nt' else 'clear')

print("""
Deep Learning con Python y Keras

Parte 6. Redes Neuronales Recurrentes

6. Práctica: Procesamiento del Lenguaje Natural

0. Contexto

En este proyecto, aprenderemos a crear un modelo generativo de texto utilizando LSTM:
* Descripción de un modelos generativos de texto.
* Enmarcar el problema de las secuencias de texto a un modelo generativo.
* Desarrollar una LSTM para generar secuencias de texto.

1. Descripción del problema: generación de texto

Vamos a trabajar el libro de "Alicia en el país de las maravillas", por lo que 
podemos descargarlo de la página (Texto sin formato UTF-8) de este libro de forma 
gratuita y colocarlo en su directorio de trabajo con el nombre 'wonderland.txt. 

Abrimos el archivo y eliminamos el encabezado:

        *** START OF THIS PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***


El pie de página es todo el texto después de la línea de texto que dice:

        THE END

2. LSTM de linea base

En esta sección desarrollaremos una red LSTM simple para aprender secuencias 
de caracteres de Alicia en el país de las maravillas. 

2.1. Cargar el dataset

Comencemos importando las clases y funciones que pretendemos usar para entrenar 
nuestro modelo.

Debemos cargar el texto ASCII y convertir todos los caracteres a minúsculas.
  
""")

# Small LSTM Network to Generate Text for Alice in Wonderland
# imports arriba

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

print("""
2.2. Conversión a numérico

No podemos modelar los caracteres directamente, sino que debemos convertir 
los caracteres a números enteros. 

Por ejemplo, la lista de caracteres en minúscula ordenados únicos en el 
libro es la siguiente:
      
""")

'''
chars = sorted(list(set(raw_text)))
set(raw_text): Convierte raw_text (que puede ser una cadena de texto) en un 
conjunto (set). 
Esto elimina automáticamente los caracteres duplicados, ya que los conjuntos 
no permiten duplicados.

list(set(raw_text)): Convierte el conjunto resultante nuevamente en una lista. 
Esto es necesario porque los conjuntos no tienen un orden definido, 
pero las listas sí.

sorted(list(set(raw_text))): Ordena alfabéticamente (o por valor Unicode) los 
elementos de la lista resultante y devuelve una nueva lista ordenada.

Resultado
El resultado es una lista ordenada que contiene todos los caracteres únicos de raw_text.

'''

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

data = list(char_to_int.items())

print("len(data)= ",len(data),"\n")
print("data[:10]=\n",data[:10],"\n") 
print("data[-10:]=\n",data[-10:],"\n") 

print("""
2.3. Dimensiones del dataset

Ahora que se cargó el libro y se preparó el mapeo, podemos resumir el conjunto de datos.

""")

# summarize the loaded data
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

print("""
En este tutorial, dividiremos el texto del libro en subsecuencias con 
una longitud fija de 100 caracteres, una longitud arbitraria. 

Cada patrón de entrenamiento de la red se compone de 100 pasos de tiempo 
de un carácter (X) seguidos de una salida de carácter (y). 

Por ejemplo, si la longitud de la secuencia es 5 (para simplificar), los dos 
primeros patrones de entrenamiento serían los siguientes:

            CHAPT -> E
            HAPTE -> R

A medida que dividimos el libro en estas secuencias, convertimos los caracteres 
a números enteros usando nuestra tabla de búsqueda que preparamos anteriormente.     
      
""")

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

print("""
Podemos ver que se han dividio en secuencias de 100
      
""")


# Secuencia 0
print("dataX[0][:10]=\n",dataX[0][:10],"\n")

# Total Secuencia 0
print("len(dataX[0]= ",len(dataX[0]),"\n")

# Para esa secuencia de X le correspondería la salida
print("dataY[0]=\n",dataY[0],"\n")

def get_key(dic,value):
        for k, v in dic.items():
                if v == value:
                        return k
        return None

# summarize the loaded data
int_to_char = dict((i, c) for i, c in enumerate(chars))

print("dataX[0] in chars=\n")
# for val in dataX[0]:
#         print(get_key(char_to_int,val),end=" ")

for val in dataX[0]:
        print(int_to_char[val],end=" ")
        
print("\n\ndataY[0] in char= ",end=" ")
print(int_to_char[dataY[0]])

print("""
2.4. Procesamiento de datos

Ahora que hemos preparado nuestros datos de entrenamiento, 
necesitamos transformarlos. 
1. Transformar la lista de secuencias de entrada en la forma 
***[muestras, pasos de tiempo, características].***
2. Cambiar la escala de los números enteros al rango de 0 a 1 (normalización) 
y usar la función de activación sigmoidea.
3. Convertir los patrones de salida (caracteres individuales convertidos en enteros) 
con One-Hot Encoding. 
      
""")

# reshape X to be [samples, time steps, features]
X=np.reshape(dataX,(n_patterns,seq_length,1))
print("X[0]=\n",X[0],"\n")
print("X[0][0]= ",X[0][0],"\b")
# normalize
X=X/float(n_vocab)
print("X[0]=\n",X[0],"\n")
# one hot encode the output variable
y=to_categorical(dataY)
print("y[0]=\n",y[0],"\n")

print("y.shape= ",y.shape,"\n")

print("""
4. LSTM más profunda

Ahora, podemos intentar mejorar la calidad del texto generado 
creando una red mucho más profunda.

1. Definimos una única capa LSTM oculta con 256 unidades de memoria. 
2. La red utiliza un Dropout del 20%. 
3. Una capa LSTM de 256 unidades.
4. Otra capa Dropout del 20%. 
4. La capa de salida es una capa densa que utiliza la función de activación Softmax. 
4. Compilación con pérdida logarítmica ('categorical_crossentropy')
5. Usaremos el algoritmo de optimización de Adam.
      
""")

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
# input_shape=(num intervalos de tiempo, num características)
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam')

print("""
2.6. Crear puntos de control

Usaremos el mejor conjunto de pesos (menor pérdida) para instanciar 
nuestro modelo generativo en la siguiente sección.
      
""")

# define the checkpoint

filepath="./pesos2/weights-improvement-{epoch:02d}-{loss:.4f}.keras"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print("""
2.7. Resultados

Utilizamos un número modesto de 20 épocas y un gran tamaño de batch 
de 128 patrones.   
      
""")

# fit the model
#model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list, verbose=2)

print("""
3. Generación de texto con LSTM

La generación de texto utilizando la red LSTM entrenada es 
relativamente sencilla.

3.1. Cargar los pesos de LSTM

En primer lugar, cargamos los datos y definimos la red 
exactamente de la misma manera

Los pesos de la red se cargan desde un archivo de punto de 
control y no es necesario entrenar la red.   
      
""")

# load the network weights
# elegimos el de menor pérdida, en este caso loss=1.9927
filename = "pesos2/weights-improvement-50-1.2026.keras"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

print("cargados los pesos de la red en su mejor versión")

print("""
3.2. Convertir de entero a carácter

Creamos un mapeo inverso que podamos usar para convertir los números 
enteros de nuevo en caracteres.
      
""")

# summarize the loaded data
# puesto arriba
#int_to_char = dict((i, c) for i, c in enumerate(chars))

print("""
3.3. Resultados y evaluación

Finalmente, necesitamos realmente hacer predicciones de manera que:
1. Comenzamos primero con una semilla como entrada
2. Generamos el siguiente carácter y 
3. Actualizamos la semilla para agregar el carácter generado al 
final y recortar el primer carácter. 

Este proceso se repite mientras queramos predecir nuevos caracteres 
(por ejemplo, una secuencia de 1000 caracteres de longitud). 
      
""")

# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("len(pattern)= ",len(pattern),"\n")
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
print("\n--------------------------------------------------\n")
# generate characters
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    # sys.stdout.write no añade retorno de carro
    # salvo que se lo pongas
    #sys.stdout.write(result)
    print(result,end="")
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone.") 

print("""
5. Mejorar nuestro modelo

A continuación, se muestra una muestra de ideas que tal vez 
desee investigar para mejorar aún más el modelo:

* Predecir menos de 1000 caracteres como salida para una semilla 
  determinada.
* Eliminar toda la puntuación del texto fuente y, por tanto, 
  del vocabulario de los modelos.
* Pruebe un One-Hot Encoding para las secuencias de entrada.
* Entrene al modelo en oraciones rellenas en lugar de secuencias 
  aleatorias de caracteres.
* Aumentar el número de épocas de entrenamiento a 100 o más.
* Agregue Dropout a la capa de entrada visible y considere ajustar 
  el porcentaje de Dropout.
* Ajuste el tamaño de batch, pruebe con un tamaño de batch de 1 
  como línea de base (muy lenta) y tamaños más grandes a partir de ahí.
* Agregue más unidades de memoria a las capas y / o más capas.
* Cambie las capas de LSTM para que tengan estado para mantener 
  el estado en todos los batch.
   
""")
   
   
