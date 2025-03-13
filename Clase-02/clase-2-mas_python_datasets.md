[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chafa618/curso_anotacion_puan/blob/main/Clase2/clase-2-mas_python_datasets.ipynb)

### Sugerencias de uso de la Notebook: 
-- Sugerimos 'Abrir en Colab' y realizar una copia del cuaderno antes de usarlo.







# 📘 Introducción a Python - Parte 2



Esta notebook cubre el uso de Python en Google Colab, que comprende el concepto de índices, bucles, condicionales hasta el uso librerías, manejo de archivos y la exploración de recursos esenciales como Kaggle HuggingFace 🤗 y GitHub (Desde el entorno de Google Colab)


## Recapitulación
- Revisar lo visto hasta acá



### Índices

Las cadenas y las listas son secuencias de elementos. O sea, conjuntos ordenados que se pueden indizar, recortar, reordenar, etc.
Es decir que podemos acceder a los elementos que componen una lista usando corchetes y un indice. Dicho indice va desde 0 a n-1, siendo n el tamaño de la lista. 


```python
lista_de_nombres = ["Gerardo", "Esteban", "Florencia", "Vanesa", "Adrian"]
type(lista_de_nombres)
```


```python
len(lista_de_nombres)
```


```python
print(lista_de_nombres[1]) #Qué pasará?
print(lista_de_nombres[:2]) #Qué pasará?
print(lista_de_nombres[2:]) #Qué pasará?
print(lista_de_nombres[0]) #Qué pasará?
print(lista_de_nombres[-1:]) #Qué pasará?
```

Asi como las cadenas de textos, las listas son elementos iterables. 
Podemos recorrer sus elementos de forma iterativa, es decir, a menudo empezamos por el principio y en cada vuelta seleccionamos un caracter, un nombre, hacemos algo con eso y continuamos hasta que se acaben los elementos o algo nos indique detenernos. 

Este recorrido se conoce como bucle. 

## 📌 Bucles


Un bucle repite un bloque de código varias veces, hasta que se cumpla una condición. Los bucles son útiles para automatizar tareas repetitivas.

Los bucles, por otra parte, existen en todos los lenguajes de programación. Aunque  difieren en su implementación y la sintaxis varía entre los distintos lenguajes, brindan funcionalidades básicas similares. 

### While:
- Tiene un **número indefinido** de iteraciones.
- La sintaxis es: `while <condición>: <bloque de código>`.
- El código del cuerpo del bucle debe 'hacer algo' hasta que la condición resulte falsa.

### For

- Se utiliza para **iterar sobre una secuencia**, como una lista, tupla o string. 
- La sintaxis es: `for <variable> in <secuencia>: <bloque de código>`. 
- El bucle itera sobre cada elemento de la secuencia. 


```python
indice = 0 

while indice < len(lista_de_nombres): # Condición del bucle

    nombre = lista_de_nombres[indice]
    print(nombre)
    indice = indice + 1
    

```


```python
x = 0 # Contador
for nombre in lista_de_nombres:
    x += 1 
    print(x, "Nombre: ",nombre)
```


```python
def regresion(n):
    while n > 0:
        print(n)
        n = n-1
    print('Despegue!')
    
regresion(int(input("Ingresá un numero:")))
```

### Expresiones booleanas

**Expresión booleana:** Es una expresión que es cierta o falsa. En Python una expresión que es verdadera tiene el valor 1 y una expresion falsa tiene el valor 0. 

Podemos generar expresiones booleanas utilizando operador de comparación u operadores relacionales. Estos comparan dos valores: **==**, **!=**, **>**, **<**, **>=** y **<=** y nos devuelven el valor de verdad del enunciado.

**ACLARACIÓN: el signo = nos permite asignar un valor a una variable. El signo == nos permite comparar dos valores (sean int, string o float)**


Otros operadores de comparación:
- x != y # x no es igual a y
- x > y # x es mayor que y
- x < y # x es menor que y
- x >= y # x es mayor o igual que y
- x <= y # x es menor o igual que y


```python
4 > 2
#4 < 2
#4 == 4   # OJO: " = "
# cuatro = 4
```

### Operadores Lógicos:


Hay tres operadores lógicos: `and`, `or`, y `not`. La semántica (significado) de estos operadores es similar a sus significados en inglés. 

Por ejemplo, `x > 0 and x <10` es verdadero **sólo si x es mayor que 0 y menor que 10**.



`n %2 == 0 or n %3 == 0` es verdadero si cualquiera de las condiciones es verdadera, o sea, si el numero es divisible por 2 o por 3.



Finalmente, el operador `not` niega una expresion booleana, de forma que `not(x > y)` es cierto si `(x > y)` es falso, o sea, si x es menor o igual que y.



```python
# Descomenta la siguiente linea para ver qué sucede
# 5 == True ?
```


### Ejecución condicional

**Sentencia condicional:** Sentencia que controla el flujo de ejecución de un programa dependiendo de cierta condición. Es decir, es una sentencia que, dependiendo de su valor de verdad, nos permite establecer si el programa se sigue ejecutando o no. La forma más simple es la sentencia **if**.

**Condición:** La expresión booleana que sucede al if en una sentencia condicional. Esta expresión determina qué rama del programa se ejecutará. 


```python
nombre = input("Cómo te llamás?")
if nombre[-1] == "a":
    print("Nombre Propio: Femenino")
```

### Ejecución alternativa: 
La ejecución alternativa de la sentencia condicional es aquella en la que tenemos más de una posibilidad. Cada condición determina qué posibilidad se ejecuta. Cada posibilidad en el flujo de la ejecución se denomina rama.

Si tenemos solamente dos posibilidades, podemos usar **if** y **else**. If define nuestra condición y else nos dice qué sucede si esa condición no es cierta.

Si queremos tener más de dos posibilidades, usamos **if**, **elif** (abreviación de 'else if') y **else**. En este caso, if define una condición, elif nos permite definir otra condición (es posible tener varios elif) y else nos dice qué sucede si nada de lo anterior es cierto.


```python
np_masc = [] # Creo dos listas vacías para separarlos
np_fem = []

for nombre in lista_de_nombres:
    if nombre[-1] == "n":
        np_masc.append(nombre) # Agrego elemento a la lista
    elif nombre[-1] == "o":
        np_masc.append(nombre) # Agrego elemento a la lista
    else:
        np_fem.append(nombre) # Agrego elemento a la lista
```


```python
np_masc
```


```python
np_fem
```

### 📍 Ejercicio:
Escribir una función que tome un carácter y devuelva True si es una vocal, de lo contrario devuelve False.

## 📌 **Uso de Librerías**

Las librerías de Python son conjuntos de funciones, clases y métodos predefinidos que extienden la funcionalidad básica del lenguaje. 
En este sentido, permiten a los desarrolladores acceder a un conjunto amplio de funcionalidades específicas, como manipulación de cadenas, operaciones matemáticas, acceso a bases de datos, manipulación de archivos, creación de interfaces gráficas, procesamiento de datos científicos, creación de sitios web, entre muchas otras.

Además las pueden encontrar como bibliotecas

### Librería estándar 

La librería estándar es un conjunto de módulos y paquetes que se distribuyen junto con Python. Muchas de las operaciones más comúnes de la programación diaria ya están implementadas en ella. 

Pueden acceder a la [documentación](https://docs.python.org/es/3.13/library/index.html)

### Cómo usamos una libreria?


```python
from string import punctuation

print(punctuation)
```


```python
import os # importamos el modulo os



os.listdir('.')   # podemos uasr la función listdir (de os)
                  # para listar los archivos y carpetas
                  # en este directorio
```


```python
from os import listdir # Importamos sólo una funcionalidad del módulo en cuestión

listdir(".") # Además podemos usar directamente la función,
             # ya que en este caso la importamos desde el 
             # módulo os
```


```python

```

## Modulos propios

Nosotros podemos construir nuestro propio set de herramientas e importar funcionalidades previamente desarolladas, a la vez que podemos hacer uso de librerías 'ajenas' o desarrolladas por alguien más, una persona, una empresa o la comunidad OS. 


```python
from mi_libreria import pasar_a_mayusculas
pasar_a_mayusculas("un texto random")
```

### 📌 Integración con Google Drive


```python
# Al ejecutar esta celda se vinculará tu cuenta de Google Drive
# Si no estás trabajando desde colab, omití este paso.

# from google.colab import drive
# drive.mount('/content/drive')
```

## 📌 Archivos

La función incorporada `open()` toma como argumento la ruta de un archivo y retorna una instancia del tipo file.

Si no se especifica una ruta, el fichero se busca en el directorio actual. Por defecto el modo de apertura es únicamente para lectura. 

La función `read()` retorna el contenido del archivo abierto.

Una vez que se ha terminado de trabajar con el fichero debe cerrarse vía `close()`.


Para abrir un archivo en modo escritura, debe especificarse en el segundo argumento.
Para escribir en él se emplea el método `write()`.

Para leer, escribir y añadir contenido de un fichero en formato binario, deben utilizarse los modos `"rb", "wb" y "ab"`, respectivamente.



```python
#Esta celda generará un "archivo_nuevo_generado_por_colab.txt" en tu almacenamiento de Drive, en caso de que estés utilizando colab.
f = open("archivo_nuevo_generado_por_colab.txt", "w") 

f.write("Hola mundo")
f.close()
```


Nótese que la función `write()` _reemplaza_ todo el contenido anterior. 

Para **añadir** datos al final del archivo sin borrar información previa, el fichero debe abrirse en la modalidad **append** `("a")`.



```python
f = open("archivo_nuevo_generado_por_colab.txt", "a")

f.write("\nHola ")
f.write("mundo")
f.close()
```

Ademas de este método, podemos usar la sentencia with open. Esto nos permite prescindir de las instancias de apertura y cierre de archivos mientras estamos trabajando.


```python
# Esta celda generará un "archivo_nuevo_generado_with_open" en tu almacenamiento de Drive
with open("archivo_nuevo_generado_with_open.txt", 'w') as file:
  archivo_modificado = file.write("linea numero 1")

```


```python
with open("archivo_nuevo_generado_with_open.txt", "r") as file2:
  print(file2.readlines())

```

## 📌 Instalación y uso de librerías externas
Colab permite instalar paquetes adicionales. Por ejemplo, podemos instalar la librería `transformers` de Hugging Face.

Si vamos a la [documentación](https://huggingface.co/docs/transformers/es/index) de la libreria nos encontramos con lo siguiente: 


"Transformers proporciona APIs para descargar y entrenar fácilmente modelos preentrenados de última generación. El uso de modelos preentrenados puede reducir tus costos de cómputo, tu huella de carbono y ahorrarte tiempo al entrenar un modelo desde cero."

"Nuestra biblioteca admite una integración perfecta entre tres de las **bibliotecas** de deep learning más populares: PyTorch, TensorFlow y JAX. Entrena tu modelo con tres líneas de código en un framework y cárgalo para inferencia con otro. Cada arquitectura de 🤗 Transformers se define en un **módulo** de Python independiente para que se puedan personalizar fácilmente para investigación y experimentos."

### Pero, pero pero....
Además de ser el nombre de una libreria o biblioteca, un transformer es un tipo de arquitectura de red neuronal, en este curso nos ocuparemos de la librería, pero no se debe confundirlas.

<div style="text-align: center;">
    <img src="imgs/autobots.jpg" alt="Autobot">
</div>



```python
# !pip install transformers
```


```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
```


```python
result = classifier("Este curso es excelente!")
print("Análisis de sentimiento:", result)
```

### Ejercicio:
- Definí una lista de 5 oraciones.
- Definí una función que tome una lista de strings y devuelva el sentiment asociado para cada una.
- Tip: Podes usar un bucle for para recorrer la lista de oraciones dentro de la función.
- Para obtener el sentiment de una oración utilizá el pipeline definido más arriba.


```python
# Escribi acá tu respuesta.

def obtener_sentiment(lista):
    return lista
```


```python

```

## 📌 Manejo de archivos en Google Colab
Podemos subir archivos directamente desde nuestra computadora y trabajar con ellos en Colab.


```python
from google.colab import files
# Subir un archivo
uploaded = files.upload() 
```


```python
# Leer un archivo CSV
import pandas as pd
df = pd.read_csv(list(uploaded.keys())[0])
print(df.head())
```

## 📌 Algunos recursos
- Una fuente de recursos es [Kaggle](https://www.kaggle.com/) acá pueden encontrar datasets, desafíos para practicar y otras yerbas.
- Otro lugar donde podemos encontrarnos con este tipo de datos es el [Hub](https://huggingface.co/datasets) de huggingFace.

### Exploración de Datasets from Hugging Face
Podemos acceder a modelos pre-entrenados y datasets.


```python
!pip install datasets
```


```python
from datasets import load_dataset
dataset = load_dataset("imdb") 


print(dataset) ## Inspeccionemos cómo luce
```

{% include copybutton.html %}
{% include additional_content.html %}


