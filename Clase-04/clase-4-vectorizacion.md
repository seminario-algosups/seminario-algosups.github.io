### Sugerencias de uso de la Notebook: 
- Sugerimos 'Abrir en Colab' y realizar una copia del cuaderno antes de usarlo.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seminario-algosups/seminario-algosups.github.io/blob/master/Clase-04/clase-4-vectorizacion.ipynb)


```python
# Para obtener el archivo de la librería que usaremos en la clase, descomentar la siguiente línea. 
# Si una no funciona, probar la otra.:
# !wget https://raw.githubusercontent.com/seminario-algosups/seminario-algosups.github.io/master/Clase-04/funciones.py
# !wget https://github.com/seminario-algosups/seminario-algosups.github.io/blob/master/Clase-04/funciones.py
```


## 🎯 **Objetivo de la clase:**
Al finalizar esta clase podrás:
- Transformar textos en vectores numéricos usando técnicas como One-Hot Encoding, Bag of Words y TF-IDF.
- Comprender cómo la representación vectorial influye en los modelos de clasificación supervisada como Naïve Bayes.
- Aplicar estas técnicas para resolver problemas prácticos de clasificación de texto.

### Pero, pero, pero...

- Retormar conceptos de la clase anterior

# Métricas de Evaluación para Modelos de Clasificación

- Métricas fundamentales para evaluar el rendimiento de un modelo de clasificación binaria.

## Precision

La precisión indica qué proporción de las predicciones positivas del modelo fueron realmente positivas.

$$
\text{Precisión} = \frac{TP}{TP + FP}
$$

Donde:
- \(TP\): Verdaderos Positivos
- \(FP\): Falsos Positivos



####  Recall (Sensibilidad)

La recall mide la capacidad del modelo para identificar correctamente todas las instancias positivas.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

Donde:
- \(FN\): Falsos Negativos



#### Accuracy (Exactitud)

La Accuracy evalúa la proporción de predicciones correctas (positivas y negativas) sobre el total.

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Donde:
- \(TN\): Verdaderos Negativos



####  F1 Score

El F1 Score es la media armónica entre la precisión y el recall. Es especialmente útil cuando los datos están desbalanceados.

$$
F_1 = 2 \cdot \frac{\text{Precisión} \cdot \text{Recall}}{\text{Precisión} + \text{Recall}}
$$




# 📌 Vectorización

## ¿Por qué vectorizar?

La vectorización es un proceso crucial en NLP ya que nos permite convertir material textual, desestructurado, en vectores numéricos. Dado que las computadoras no pueden procesar directamente texto en su forma original y que los modelos de Machine Learning trabajan con datos **numéricos y estructurados**, necesitamos traducir de cierta forma ese input a valores que las computadoras pueden procesar.



### **Ejemplo:**
Supongamos que queremos entrenar un modelo para detectar si una reseña de una película es positiva o negativa.

 **Entrada:**  
*"Me encantó la película, la actuación fue brillante."*

 **Lo que la computadora entiende:** 

🚫 `"Me encantó la película, la actuación fue brillante."`  

✅ Vector numérico: `[0.2, 0.4, 0.7, 0.1, ...]`

El texto necesita convertirse en **una representación matemática**, y el proceso de convertir palabras en números se llama **vectorización**.





### Vectores:
Los vectores son básicamente conjuntos de números que representan diversas características del texto. Estos conjuntos pueden tener distintas dimensiones:

- Vectores 1D : representan palabras individuales (por ejemplo, word embeddings).
- Vectores 2D : representan secuencias de palabras, como oraciones o documentos (por ejemplo, sentence embeddings).
- Vectores multidimensionales : pueden representar estructuras y relaciones más complejas, involucrando potencialmente espacios de dimensiones superiores.

Al aplicar diferentes técnicas de vectorización, los vectores resultantes variarán según el método utilizado. Cada técnica produce vectores con características y rangos de valores únicos. 

Por ejemplo, algunas técnicas producen valores binarios (0 o 1), mientras que otras producen valores continuos entre 0 y 1. 



```python
# En Python podemos pensar que un vector es una lista de números
# TODO: Verificar que sea necesario aclarar esto, a esta altura del partido es confundirlos aun mas... 
import numpy as np

vector = [1, 2, 3, 4, 5]
otro_vector = [[0.1, 0.2, 0.8, -0.3, 0.5]]

np_vector = np.array(vector)
np_otro_vector = np.array(otro_vector)


print(np_vector)
print(np_vector.shape) # (5,) -> 5 elementos en una dimensión (1D vector)

print(np_otro_vector)
print(np_otro_vector.shape) # (1, 5) -> 1 fila con 5 elementos (2D vector). Sería una matriz de 1x5 en lugar de un vector de 5 elementos.
```




## 📌 **El desafío de representar el texto**
Al vectorizar texto, buscamos representar su significado semántico en forma numérica. Esto implica preservar al máximo los matices y el contexto de las palabras, permitiendo que los modelos los interpreten y utilicen de manera efectiva.


| Característica | Desafío |
|--------------|---------|
| El lenguaje es ambiguo | "Banco" puede referirse a una entidad financiera o un asiento. |
| Orden de palabras importa | "El perro mordió al hombre" vs. "El hombre mordió al perro". |
| Las palabras tienen significado | "Perro" y "can" son similares, pero el modelo debe aprender esto. |
| Vocabulario extenso | ¿Cómo representar millones de palabras en un espacio finito? |



📍 Ejemplo : Imaginemos que tenemos dos oraciones:

- "El perro se sienta en la alfombra".
- "Un can descansa sobre tapete".


Sin la vectorización, un modelo de ML vería estos strings de formas completamente diferentes porque utilizan palabras diferentes. Pero con vectorización, especialmente con técnicas avanzadas como *word embeddings*, el modelo puede entender que "perro" y "can", así como "alfombra" y "tapete", tienen significados similares.

Supongamos que usamos una técnica de vectorización avanzada que nos permite hacer lo siguiente:

- "perro" podría convertirse en un vector como `[0,2, 0,5, 0,1, …]`
- "can" podría convertirse en un vector como `[0,21, 0,49, 0,12, …]`


En este caso, los vectores de "perro" y "can" estarían muy próximos en el espacio vectorial, lo que indica su similitud semántica. Lo mismo ocurre con "tapete" y "alfombra".


Esto significa que debemos elegir **formas de vectorización** que conserven **información clave** sin hacer el problema computacionalmente inmanejable.



## **Distintos métodos de vectorización**
Existen distintos métodos de vectorización, que van desde **técnicas simples** hasta **modelos más avanzados**, con sus complejidades particulares:

### 🔹 Representaciones básicas:
1. **One-Hot Encoding:** 
- Representacion binaria, indica precencia o ausencia de una palabra.
- Suele usarse en contextos generales sobre variables categóricas.
- Escala muy mal > Se genera un vector de dimensión igual al número de palabras. 

2. **Bag of Words:** 
- Representa frecuencia de palabras.
- A diferencia del OHE, no sólo indica si la palabra está presente sino cuántas veces aparece.
- No considera la posición de las palabras ni el contexto.
- No tiene en cuenta la semántica

3. **TF-IDF (Term Frequency - Inverse Document Frequency):** 
- Parte de la idea de BOW pero ajusta la importancia de palabras comunes en el corpus.
- Reduce el peso de las palabras muy frecuentes (como stopwords), a la vez que aumenta la importancia de las menos frecuentes (distintivas para un documento).

### 🔹 Representaciones más avanzadas:
4. **Word Embeddings (Word2Vec, GloVe, FastText):** 
- Capturan la semántica de las palabras y sus relaciones en un espacio vectorial de menor dimensión.
- Permiten capturar similitudes entre palabras a partir de su distancia/cercanía en el espacio vectorial.

5. **Representaciones Contextuales (BERT, GPT):** 
- Extienden la capacidad de los _embeddings_ teniendo en cuenta el contexto completo en el que aparece una palabra.
- Cada palabra el documento tiene una representacion distinta según las palabras que la rodean. 




Para esta clase, nos enfocaremos en **las primeras tres técnicas** y cómo usarlas en un **Naïve Bayes Classifier** para clasificación de texto.


## 📌 Bag of Words
Para preparar el terreno, vamos a ver un ejemplo sencillo de cómo un texto puede representarse numéricamente.

**Texto original:**
- "El gato duerme en la cama"
- "El perro ladra en la calle"

 **Vocabulario único (Bag of Words):**  

- ["el", "gato", "duerme", "en", "la", "cama", "perro", "ladra", "calle"]

 **Matriz de conteo:**

| Documento | el | gato | duerme | en | la | cama | perro | ladra | calle |
|-----------|----|------|--------|----|----|------|-------|-------|-------|
| "El gato duerme en la cama" | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| "El perro ladra en la calle" | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 1 |

Cada fila representa un documento y cada columna una palabra del vocabulario. 

**Esto es lo que los modelos de Machine Learning ven** en lugar del texto original.



```python
# !pip install numpy scikit-learn matplotlib nltk 
```


```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
```


```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import re

stop_words = set(stopwords.words('spanish'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text, language='spanish')
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

```


```python

corpus = ["El gato duerme en la cama", "El perro ladra en la calle", "Un gato descansa en el tapete", "Los perros juegan en la calle", "El gato y el perro juegan."]
norm_corpus = [preprocess_text(doc) for doc in corpus]

tokens = [word for doc in norm_corpus for word in word_tokenize(doc)]
fdist = FreqDist(tokens)
print(fdist.most_common()) # Pasarle n como argumento para limitar la cantidad de palabras a mostrar

```


```python
from sklearn.feature_extraction.text import CountVectorizer

cv_vectorizer = CountVectorizer()
X_cv_vectorizer = cv_vectorizer.fit_transform(norm_corpus)

```


```python
print("Matriz Termino documento")
print(cv_vectorizer.get_feature_names_out())  # Ver palabras en vocabulario
print("\n", X_cv_vectorizer.toarray())  # Matriz de términos
```


```python
from funciones import plot_text_word_matrix

plot_text_word_matrix(norm_corpus)
```


```python
from funciones import plot_bag_of_words, plot_tfidf_matrix, plot_text_word_matrix

#plot_bag_of_words(corpus)
```


```python

```


**🔹 ¿Qué problema tiene este enfoque?**
- No considera el orden de las palabras.
- No diferencia palabras con significados similares.
- Puede generar vectores muy grandes y dispersos.



## 📌 TF-IDF (Term Frequency - Inverse Document Frequency)

*One of the earliest methods of text representation that laid the groundwork for subsequent advances was Term Frequency–Inverse Document Frequency (TF–IDF). This method quantifies the importance of a word within a document relative to a corpus by accounting for term frequency and inverse document frequency*

Uno de los primeros métodos de representación textual que sentó las bases para los siguientes avances es TF-IDF, por sus siglas en ingés. Esta técnica 'cuantifica' la importancia de una palabra en un documento en relación con un corpus teniendo en cuenta la **frecuencia de términos y la frecuencia inversa de documentos.

Punto clave: Ajusta la frecuencia de las palabras para evitar que términos comunes dominen la representación.

Conceptos clave:

- TF (Term Frequency): Frecuencia normalizada de una palabra en un documento.
- IDF (Inverse Document Frequency): Reduce la importancia de palabras muy frecuentes en todos los documentos.

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
$$

$$
\text{TF}(t, d) = \frac{\text{Número de veces que } t \text{ aparece en } d}{\text{Número total de términos en } d}
$$

$$
\text{IDF}(t) = \log \left( \frac{N}{n_t} \right)
$$

🔹 Ventaja: Mejora la representación considerando la relevancia de palabras. 


🔹 Desventaja: No captura la semántica de las palabras.




```python
from sklearn.feature_extraction.text import TfidfVectorizer
```


```python

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(norm_corpus)


```


```python
print("Matriz de TF-IDF:")
print(X_tfidf.toarray())
print(tfidf_vectorizer.get_feature_names_out())
```


```python
from funciones import plot_tfidf_matrix

plot_tfidf_matrix(X_tfidf, norm_corpus, tfidf_vectorizer.get_feature_names_out())
```

## Aplicacion de vectorización en un clasificador

- Aplicar técnicas básicas de limpieza y normalización de datos.
- Aplicar técnicas básicas de vectorización sobre los datos.
- Alimentar un modelo de ML para una tarea específica de NLP


```python

```


```python

import pandas as pd

csv_data = pd.read_csv("synth_data_sentiment.csv")

#csv_data.head()
#csv_data.info()
#len(cvs_data)
```


```python
## Limpieza de los datos

# Eliminar filas con valores nulos
csv_data = csv_data.dropna()
# Eliminar filas duplicadas
csv_data = csv_data.drop_duplicates()
# Eliminar filas con texto vacío
csv_data = csv_data[csv_data['text'].str.strip() != '']

# Normalizar nuestros documentos
#csv_data['norm_text'] = csv_data['text'].apply(preprocess_text)
```


```python
csv_data.head()
```


```python
sentiment_corpus = csv_data["text"].to_list() #  
sentiment_labels = csv_data["label"].tolist()  # 1 = positivo, 0 = negativo
```


```python
# Podemos comenzar x revisar la nube de palabras, o directamente la matriz. 
# Matriz dispersa
plot_bag_of_words(sentiment_corpus)
```


```python
from funciones import plot_text_word_matrix, plot_tfidf_matrix

# plot_bag_of_words(sentiment_corpus)
```


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
def vectorize_corpus(vectorizer, corpus, preprocess=True):
    """
    Vectoriza un corpus de texto utilizando el vectorizer requerido.
    """
    if vectorizer == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words=list(stop_words))
    elif vectorizer == 'count': 
        vectorizer = CountVectorizer()
    else: 
        raise ValueError("El vectorizer debe ser 'tfidf' o 'count'")
    
    # Si el corpus es una lista de documentos, vectorizamos cada uno
    # Si el corpus es un solo documento, lo vectorizamos directamente
    if isinstance(corpus, list):
        corpus = [preprocess_text(doc) for doc in corpus] if preprocess else [doc for doc in corpus]
    else:
        corpus = preprocess_text(corpus) if preprocess else corpus   
    
    # Vectorizamos el corpus
    X = vectorizer.fit_transform(corpus)
    
    return X, vectorizer
```


```python
# 1. Utilicemos countVectorizer como ejemplo

X_count, count_vectorizer = vectorize_corpus('count', sentiment_corpus, preprocess=False)
print("Matriz de conteo:")
print(X_count.toarray()[0])
print("\nVocabulario (50):") 
print(count_vectorizer.get_feature_names_out()[:50])

```


```python
# 2. Ahora utilizamos tfidfVectorizer

X_tfidf, vectorizer_tfidf = vectorize_corpus('tfidf', sentiment_corpus, preprocess=False)
```


```python
plot_tfidf_matrix(X_tfidf, sentiment_corpus, vectorizer_tfidf.get_feature_names_out()) 
```


```python
## Como se ve, acá no es muy posible ver la matriz, ya que es muy grande y aperentemente no tiene sentido.
```


```python

```

## Reducción de dimensionalidad
La matriz TF-IDF es grande y dispersa; aunque podríamos intentar reducir el numero de *features* que tomará el vectorizador, a los objetivos de la clase resulta más interesante pasar por un algoritmo sencillo de reducción de dimensionalidad. 

Para representar la matrix visualmente necesitamos reducirla a dos dimensiones:

🔹 **UMAP** (*Uniform Manifold Approximation and Projection*)

- Preserva relaciones locales y globales mejor que t-SNE.

- Muy útil en visualizaciones de alta calidad para embeddings textuales.


### Caracteristicas
- **Construcción de un grafo de vecinos**: 
 - se calcula una representación de los datos mediante un grafo en el que cada punto de datos está conectado a sus vecinos más cercanos. 
 UMAP calcula la probabilidad de que un punto esté conectado a otro en función de la distancia entre ellos en el espacio original de alta dimensión.
- **Optimización de la proyección:**
 Luego, UMAP proyecta los datos de alta dimensión a un espacio de baja dimensión, manteniendo la estructura del grafo de vecinos. 


```python
# !pip install umap-learn
```


```python
import umap.umap_ as umap

umap_reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
X_embedded = umap_reducer.fit_transform(X_tfidf.toarray())
```


```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
plt.scatter(X_embedded[:,0], X_embedded[:,1], s=5, alpha=0.7)
plt.title("Visualización UMAP de vectores TF-IDF", fontsize=14)
plt.xlabel("UMAP Dimensión 1")
plt.ylabel("UMAP Dimensión 2")
plt.grid(True)
plt.show()
```


```python
# !pip install plotly
```


```python
import plotly.express as px
import pandas as pd

df_plot = pd.DataFrame({
    "dim1": X_embedded[:, 0],
    "dim2": X_embedded[:, 1],
    "texto": sentiment_corpus
})

fig = px.scatter(df_plot, x='dim1', y='dim2', hover_data=['texto'],
                 title='UMAP interactivo de vectores TF-IDF')
fig.show()
```


```python
from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt

tfidf_mean = np.asarray(X_tfidf.mean(axis=0)).flatten()
terms = vectorizer_tfidf.get_feature_names_out()
tfidf_scores = dict(zip(terms, tfidf_mean))

wordcloud = WordCloud(width=800, height=400, background_color='white')
wordcloud.generate_from_frequencies(tfidf_scores)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de palabras promedio TF-IDF', fontsize=14)
plt.show()
```

### Naive Bayes


```python

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```


```python

# Dividimos los datos en entrenamiento y prueba
# La función train_test_split divide los datos en conjuntos de entrenamiento y prueba
# Por ahora vamos a utilizarla dos veces, puesto que tenemos dos X uno con tfidf y otro con count vectorizer
```


```python
# MNB + vectorización con count vectorizer

X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_count, sentiment_labels, test_size=0.3, random_state=42)
model_wiht_cv = MultinomialNB()
model_wiht_cv.fit(X_train_cv, y_train_cv)


```


```python
# Evaluamos el modelo
y_pred_cv = model_wiht_cv.predict(X_test_cv)
print(f"Accuracy: {accuracy_score(y_test_cv, y_pred_cv):.2f}")
```


```python
# NB + vectorización con tfidf vectorizer
# Entrenamos el modelo

X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(X_tfidf, sentiment_labels, test_size=0.3, random_state=42)
model = MultinomialNB()
model.fit(X_train_tf, y_train_tf)

# Evaluamos el modelo
y_pred_tf = model.predict(X_test_tf)
print(f"Accuracy: {accuracy_score(y_test_tf, y_pred_tf):.2f}")
```


```python
nueva_sentence = "todo muy lindo salvo por la comida que era horrible"
print(model.predict(vectorizer_tfidf.transform([nueva_sentence])))

if model.predict(vectorizer_tfidf.transform([nueva_sentence])) == 1:
    print("Sentimiento positivo")
else: 
    print("Sentimiento negativo")
```


```python
nueva_sentence = "todo muy lindo salvo por la comida que era horrible"
print(model_wiht_cv.predict(count_vectorizer.transform([nueva_sentence])))

if model_wiht_cv.predict(count_vectorizer.transform([nueva_sentence])) == 1:
    print("Sentimiento positivo")
else: 
    print("Sentimiento negativo")

```


```python
def predict_sentiment(model, vectorizer, sentence):
    """
    Predice el sentimiento de una oración dada utilizando el modelo y el vectorizador proporcionados.
    """
    preprocessed_sentence = preprocess_text(sentence)
    prediction = model.predict(vectorizer.transform([preprocessed_sentence]))
    return prediction[0]  # 1 = positivo, 0 = negativo
```


```python
print(predict_sentiment(model, vectorizer_tfidf, "La pelicula es malisima, a pesar de los buenos dialogos."))

print(predict_sentiment(model_wiht_cv, count_vectorizer, "La pelicula es malisima, a pesar de los buenos dialogos."))
```


```python
testset_df = pd.read_csv("synth_eval_sentiment.csv")
```


```python
print(testset_df)

```


```python
#testset_df['norm_text'] = testset_df['text'].apply(preprocess_text)
testset_df['predicted_label_tfidf'] = testset_df['text'].apply(lambda x: predict_sentiment(model, vectorizer_tfidf, x))
```


```python
testset_df['predicted_label_cv'] = testset_df['text'].apply(lambda x: predict_sentiment(model_wiht_cv, count_vectorizer, x))
```


```python
!pip install seaborn
```


```python
from sklearn.metrics import classification_report   
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.xticks(ticks=[0.5, 1.5], labels=['Negativo', 'Positivo'])
    plt.yticks(ticks=[0.5, 1.5], labels=['Negativo', 'Positivo'], rotation=0)
    plt.show()
    
def make_classification_report(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred)
    print(f"Reporte de clasificación - {model_name}")
    print(report)
```


```python
plot_confusion_matrix(testset_df['label'], testset_df['predicted_label_tfidf'], "TF-IDF")
```


```python
make_classification_report(testset_df['label'], testset_df['predicted_label_tfidf'], "TF-IDF")
```


```python
# Ahora veamos los resultados del Count Vectorizer

plot_confusion_matrix(testset_df['label'], testset_df['predicted_label_cv'], "Count Vectorizer")
make_classification_report(testset_df['label'], testset_df['predicted_label_cv'], "Count Vectorizer")
```


```python

```


```python

```
