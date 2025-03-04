<a href="https://colab.research.google.com/gist/chafa618/886efbd6e21e4037cb5f7b9676fe94cd/clase-1-introducci-n-a-python.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## 📘 Introducción a la programación en Python - Parte 1

Esta notebook cubre la introducción al curso, una presentación de sintaxis básica de python y su uso en google Colab

📌 *¿Qué es programar?*

**Solución de problemas:** El proceso de formular un problema, hallar la solución y expresar esa solución.


**Programa:** Un conjunto de instrucciones que especifica una computación.


**Algoritmo:** Un proceso general para resolver una clase completa de problemas.


**programación:** El proceso de romper una tarea en tareas cada vez más pequeñas hasta que puedan ser ejecutadas con una de estas instrucciones simples.



## 📌 Lenguaje Formal vs. Lenguaje Natural



**lenguaje natural:** Cualquier lenguaje hablado que evolucionó de forma natural. Español, inglés, etc...




**lenguaje formal:** Cualquier lenguaje diseñado por humanos que tiene un propósito específico, como la representación de ideas matemáticas o programas de computadoras; todos los lenguajes de programación son lenguajes formales.


> **semántica**: El significado de un programa.


> **sintaxis**: La estructura de un programa.


> **unidad**: Uno de los elementos básicos de la estructura sintáctica de un programa, análogo a una palabra en un lenguaje natural.


> **análisis sintáctico:** La examinación de un programa y el análisis de su estructura sintáctica.

## 📌 ¿Por qué Python?

**Python**: Lenguaje de alto nivel. Muy usado actualmente (con muchos recursos disponibles). Portable.


__Lenguaje de alto nivel:__ Lenguaje diseñado para ser fácil de leer y escribir para la gente. La computadora debe traducirlo a un lenguaje de bajo nivel para entenderlo. Ej: Python.


_Lenguaje de bajo nivel:_ Lenguaje diseñado para ser fácil de ejecutar para una computadora; también “lenguaje de máquina” o “lenguaje ensamblador”. Ej: Código binario.


_Lenguaje de nivel medio:_ Utilizan estructuras típicas de los lenguajes de alto nivel pero, a su vez, permiten un control a muy bajo nivel. Ej: C.


__Portabilidad:__ La cualidad de un programa que le permite ser ejecutado en más de un tipo de computadora.



### 📌 Algunos conceptos básicos




**Variable:** nombre que hace referencia a un valor. A diferencia de los strings, no lleva comillas.


**Valor:** un número o cadena (o cualquier otra cosa que se especifique posteriormente) que puede almacenarse en una variable o calcularse en una expresión.
_Ej: edad = 17 (Variable: edad. Valor: 17)_


**Sentencia:** es una porción de código que representa una orden o acción. 

- Asignación: Sentencia que asigna un valor a una variable.
 - _Ejemplo:_ edad = 17



**operador:** un símbolo especial que representa un cálculo sencillo, como la suma (1+1), la multiplicación (1*1) o la concatenación de cadenas (“Hola, ”+nombre).

**expresión:** una combinación de variables, operadores y valores. Dicha combinación representa un único valor como resultado. 

### 📌 Variables y Asignación


```python

a = 10  # Cambia estos valores
b = 5   # Cambia estos valores
```

    9



```python
print(a)
```


```python

print("Suma:", a + b)
print("Resta:", a - b)
print("Multiplicación:", a * b)
print("División:", a / b)
```


```python
concatenar = "hola, " + "amigos"
print(concatenar)
```

    hola, amigos



```python
nombre = "Fernando" # Variable nombre: Fernando
print("Nombre: ",nombre) # Instrucción: imprimir string 'Nombre:' y la variable nombre 

edad = 27 
print(nombre, " tiene ", edad, " años.") 
```

    Nombre:  Fernando
    Fernando  tiene  27  años.



#### 📌 Funciones:
Son instrucciones que Python reconoce y ejecuta. Las funciones por lo regular
tienen como resultado un valor que se puede guardar en una variable. Las funciones pueden **tomar parámetros** o no, y pueden **devolver un resultado** o no.




```python
#Funcion con un parametro
def saludar(name): ## Parámetro: name (interno a la función)
    print("hola, ", name)

saludar(nombre) ## Argumento: la cadena "Fernando" almacenada en la variable 'nombre'.
```

    hola,  Fernando



```python
#Funcion con dos parametros
def datos(name, age):
    print("Nombre: ", name)
    print("Edad: ", age)
    
datos(nombre, edad)


```

    Nombre:  Fernando
    Edad:  27


### 📌 Tipos de datos


Para verificar el tipo de dato Python ofrece la función nativa type() que devuelve el tipo de dato de aquello que le pasamos como argumento. Hay varios tipos de datos, pero nombramos acá solo algunos a fines de ejemplos.

Hay que tener en cuenta que los distintos tipos de dato tienen distintos 'comportamientos'. Si por alguna razón pedimos que python realice un corportamiento típico de un tipo de dato, pero le pasamos otro tipo de dato obtendremos un error. Veremos esto un poco más adelante


```python
print(type(nombre))
print(type(edad))
```

    <class 'str'>
    <class 'int'>


| tipo | descripción | ejemplo |
| :---------: | :-----: | :------: |
| int | Entero | 3 |
| float | flotante  | 3.0 |
| str  | string | "3" |
| bool  | booleano | True o False |

**📍 Conversiones de tipos de datos**

| descripción | función | devuelve |
| :---------: | :-----: | :------: |
| Convierte en entero | int(20.9) | 30 |
| Convierte en string | str(402)  | “402” |
| Convierte en float  | float(40) | 40.0 |
| Convierte en lista | list(40) | [40] 

### 📌 String

Las strings en python son un tipo inmutable que permite almacenar secuencias de caracteres y se indican entre comillas (simples o dobles), por lo tanto "Hola" y 'Hola' son equivalentes. Esto no es así en todos los lenguajes. 

Las cadenas no están limitadas en tamaño, por lo que el único límite es la memoria de tu ordenador. Una cadena puede estar también vacía.

Como cualquier tipo de dato, los strings tienen métodos que les son propios



```python
nombre.upper() #Convertir a mayúsculas
```




    'FERNANDO'




```python
nombre.lower() #A minúsculas
```




    'fernando'




```python
len(nombre) #Largo, pueden probar con otros datos cambiando el valor o directamente la variable 

```




    8




```python
list(nombre)
```




    ['F', 'e', 'r', 'n', 'a', 'n', 'd', 'o']




```python
int(nombre) # Qué pasa si quiero convertir a entero un string? 
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-7-f7e1ef64f59a> in <module>()
    ----> 1 int(nombre)
    

    ValueError: invalid literal for int() with base 10: 'Fernando'



```python
x = 5
s = "El número es: " + str(x)
print(s) #El número es: 5
```

### Listas

Las listas en Python son una estructura o tipo de dato que permite almacenar otros datos de cualquier tipo. Son mutables y dinámicas, lo cual es la principal diferencia con los sets y las tuplas.


📍 Algunas propiedades de las listas:

- Son ordenadas, mantienen el orden en el que han sido definidas
- Pueden ser formadas por tipos arbitrarios
- Pueden ser indexadas con [i].
- Se pueden anidar, es decir, meter una dentro de la otra.
- Son mutables, ya que sus elementos pueden ser modificados.
- Son dinámicas, ya que se pueden añadir o eliminar elementos.


```python
nombre_completo = "Juan Gimenez"
nombre_apellido = nombre_completo.split(' ')
```


```python
print(nombre_completo)
print(nombre_apellido)
```

    Juan Gimenez
    ['Juan', 'Gimenez']



**.split** como vemos, es un método que separa los strings. Y el método inverso es **.join**, para unir muchas cadenas intercalandolas con otra


```python
" ".join(nombre_apellido)
```




    'Juan Gimenez'




```python
nombre_apellido.pop() 
```




    True



### Índices

Las cadenas y las listas son secuencias de elementos. O sea, conjuntos ordenados que se pueden indizar, recortar, reordenar, etc.
Es decir que podemos acceder a los elementos que componen una lista usando corchetes y un indice. Dicho indice va desde 0 a n-1, siendo n el tamaño de la lista. 



```python
lista_de_nombres = ["Gerardo", "Esteban", "Florencia", "Vanesa", "Adrian"]
type(lista_de_nombres)
```




    list




```python
len(lista_de_nombres)
```




    5




```python
print(lista_de_nombres[1]) #Qué pasará?
print(lista_de_nombres[:2]) #Qué pasará?
print(lista_de_nombres[2:]) #Qué pasará?
print(lista_de_nombres[0]) #Qué pasará?
print(lista_de_nombres[-1:]) #Qué pasará?
```

    Esteban
    ['Gerardo', 'Esteban']
    ['Florencia', 'Vanesa', 'Adrian']
    Gerardo
    ['Adrian']



```python

```

## 📌 Bucles


```python
print([x for x in lista_de_nombres])
```

    ['Gerardo', 'Esteban', 'Florencia', 'Vanesa', 'Adrian']



```python
x = 0 
for nombre in lista_de_nombres:
    x += 1 # Contador
    print(x, "Nombre: ",nombre)
```

    1 Nombre:  Gerardo
    2 Nombre:  Esteban
    3 Nombre:  Florencia
    4 Nombre:  Vanesa
    5 Nombre:  Adrian



```python
def regresion(n):
    while n > 0:
        print(n)
        n = n-1
    print('Despegue!')
    
regresion(int(input("Ingresá un numero:")))
```

    Ingresá un numero:4
    4
    3
    2
    1
    Despegue!


### Expresiones booleanas

**Expresión booleana:** Es una expresión que es cierta o falsa. Podemos generar expresiones booleanas utilizando operador de comparación u operadores relacionales. Estos comparan dos valores: **==**, **!=**, **>**, **<**, **>=** y **<=** y nos devuelven el valor de verdad del enunciado.

**ACLARACIÓN: el signo = nos permite asignar un valor a una variable. El signo == nos permite comparar dos valores (sean int, string o float)**


```python
4 > 2
#4 < 2
#4 == 4   # OJO: " = "
# cuatro = 4
```




    True




### Ejecución condicional

**Sentencia condicional:** Sentencia que controla el flujo de ejecución de un programa dependiendo de cierta condición. Es decir, es una sentencia que, dependiendo de su valor de verdad, nos permite establecer si el program se sigue ejecutando o no. La forma más simple es la sentencia **if**.

**Condición:** La expresión booleana que sucede al if en una sentencia condicional. Esta expresión determina qué rama del programa se ejecutará. 


```python
nombre = input("Cómo te llamás?")
if nombre[-1] == "a":
    print("Nombre Propio: Femenino")
```

    Ingresá un número: 5
    Elegiste un numero positivo


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




    ['Gerardo', 'Esteban', 'Adrian']




```python
np_fem
```




    ['Florencia', 'Vanesa']



### 📍 Ejercicio:
Escribir una función que tome un carácter y devuelva True si es una vocal, de lo contrario devuelve False.



```python

```

### 📌 Integración con Google Drive


```python
# Al ejecutar esta celda se vinculará tu almacenamiento en drive. 
#Deberás acceder al link para autorizar la integración

from google.colab import drive
# This will prompt for authorization.
drive.mount('/content/drive')
```

    Mounted at /content/drive


## 📌 Archivos

La función incorporada open() toma como argumento la ruta de un archivo y retorna una instancia del tipo file.
Si no se especifica una ruta, el fichero se busca en el directorio actual. Por defecto el modo de apertura es únicamente para lectura. La función read() retorna el contenido del archivo abierto.
Una vez que se ha terminado de trabajar con el fichero debe cerrarse vía close().


Para abrir un archivo en modo escritura, debe especificarse en el segundo argumento.
Para escribir en él se emplea el método write().

Para leer, escribir y añadir contenido de un fichero en formato binario, deben utilizarse los modos "rb", "wb" y "ab", respectivamente.



```python
#Esta celda generará un "archivo_nuevo_generado_por_colab.txt" en tu almacenamiento de Drive
f = open("archivo_nuevo_generado_por_colab.txt", "w") 

f.write("Hola mundo")
f.close()
```

Nótese que la función write() reemplaza todo el contenido anterior. Para añadir datos al final del archivo sin borrar información previa, el fichero debe abrirse en la modalidad append ("a").



```python
f = open("archivo_nuevo_generado_por_colab.txt", "a")

f.write("\nHola ")
f.write("mundo")
f.close()
```


```python
Para leer, escribir y añadir contenido de un fichero en formato binario, deben utilizarse los modos "rb", "wb" y "ab", respectivamente.

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

    ['linea numero 1']



```python

```
