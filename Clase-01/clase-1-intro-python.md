<a href="https://drive.google.com/file/d/1guwe2a-cyMfmOaVHNXc7gTbJw2Rx65fR/view?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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


__Lenguaje de alto nivel:__ Lenguaje diseñado para ser fácil de leer y escribir para la gente. La computadora debe traducirlo a un lenguaje de bajo nivel para entenderlo. Ej: Python, C++, Java.


__Lenguaje de bajo nivel:__ Lenguaje diseñado para ser fácil de ejecutar para una computadora; también “lenguaje de máquina” o “lenguaje ensamblador”. Ej: Código binario.

__Portabilidad:__ La cualidad de un programa que le permite ser ejecutado en más de un tipo de computadora. Los lenguajes de alto nivel son más portables que los de bajo nivel.

Existen dos tipos de programas para traducir lenguajes de alto nivel a lenguajes de bajo nivel: 

- *Intérpretes*: Programa que traduce y ejecuta progresivamente un lenguaje de alto nivel.
- *Compiladores*: Programa que traduce todo un código fuente en un lenguaje de alto nivel a un código ejecutable de bajo nivel que luego se ejecuta mediante un ejecutor.

Python es un lenguaje con intérprete.

### 📌 Algunos conceptos básicos



**Variable:** nombre que hace referencia a un valor. A diferencia de los strings, no lleva comillas.


**Valor:** un número o cadena (o cualquier otra cosa que se especifique posteriormente) que puede almacenarse en una variable o calcularse en una expresión.
_Ej: edad = 17 (Variable: edad. Valor: 17)_

**tipo:** Los lenguajes formales dividen los valores en distintas clases. Cada una de estas clases tiene características particulares y los operadores son sensibles a ellas. En Python, los tipos incluyen a los enteros (*integer*), los decimales (*float*), las cadenas (*strings*), los conjuntos (*sets*) y los diccionarios.

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


```python
nombre = "Fernando" # Variable nombre: Fernando
print("Nombre: ",nombre) # Instrucción: imprimir string 'Nombre:' y la variable nombre 

edad = 27 
print(nombre, " tiene ", edad, " años.") 
```


#### 📌 Funciones:
Son instrucciones que Python reconoce y ejecuta. Las funciones por lo regular
tienen como resultado un valor que se puede guardar en una variable. Las funciones pueden **tomar parámetros** o no, y pueden **devolver un resultado** o no.




```python
#Funcion con un parametro
def saludar(name): ## Parámetro: name (interno a la función)
    print("hola, ", name)

saludar(nombre) ## Argumento: la cadena "Fernando" almacenada en la variable 'nombre'.
```


```python
#Funcion con dos parametros
def datos(name, age):
    print("Nombre: ", name)
    print("Edad: ", age)
    
datos(nombre, edad)


```

### 📌 Tipos de datos


Para verificar el tipo de dato Python ofrece la función nativa type() que devuelve el tipo de dato de aquello que le pasamos como argumento.

Hay que tener en cuenta que los distintos tipos de dato tienen distintos 'comportamientos'. Si por alguna razón pedimos que python realice un corportamiento típico de un tipo de dato, pero le pasamos otro tipo de dato obtendremos un error. Veremos esto un poco más adelante


```python
print(type(nombre))
print(type(edad))
```

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


```python
nombre.lower() #A minúsculas
```


```python
len(nombre) #Largo, pueden probar con otros datos cambiando el valor o directamente la variable 

```


```python
list(nombre)
```


```python
int(nombre) # Qué pasa si quiero convertir a entero un string? 
```


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


**.split** como vemos, es un método que separa los strings. Y el método inverso es **.join**, para unir muchas cadenas intercalandolas con otra


```python
" ".join(nombre_apellido)
```


```python
nombre_apellido.pop() 
```

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


```python

```

{% include copybutton.html %}
{% include additional_content.html %}
