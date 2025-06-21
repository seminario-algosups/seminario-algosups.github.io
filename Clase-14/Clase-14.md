### Sugerencias de uso de la Notebook: 
- Sugerimos 'Abrir en Colab' y realizar una copia del cuaderno antes de usarlo.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seminario-algosups/seminario-algosups.github.io/blob/master/Clase-14/clase-14.ipynb)


```python
# Importaciones necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from pysentimiento.preprocessing import preprocess_tweet

print("Bibliotecas importadas correctamente.")
```

## Carga de datos
Primero, dividimos nuestros datos en un conjunto de entrenamiento (para que el modelo aprenda) y un conjunto de prueba (para evaluarlo de forma objetiva).


```python
from datasets import load_dataset

# Cargar el dataset desde Hugging Face
dataset = load_dataset("mrm8488/tass-2019")

df = pd.DataFrame(dataset["train"])
# df_test = pd.DataFrame(dataset["test"])

df = df[["sentence", "labels"]].dropna()
#df_test = df_test[["sentence", "labels"]].dropna()

print("Dataset de ejemplo:")
print(df)
```


```python

```


```python
# Ahora limpiamos el df para quedarnos solo con las labels 0 (neg) y 1 (pos) 
clean_df_train = df[df['labels'].isin([0, 1])]
clean_df_train["labels"].unique()
print(len(clean_df_train))
```

## Normalización

Como preprocesamiento vamos a aplicarle la función de Pysentimiento "preprocess_tweet" a nuestro dataset.




```python
# Normalizamos los tweets unsando pysentimiento
clean_df_train["sentence"] = clean_df_train["sentence"].apply(preprocess_tweet)
```


```python

# Acá hacemos un poco de trampa.
# Duplicamos el dataset para ampliar la cantidad de datos y mejorar performance

augmented_ds = pd.concat([clean_df_train, clean_df_train], ignore_index=True)
print(len(augmented_ds))
```


```python
# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    augmented_ds['sentence'], augmented_ds['labels'], test_size=0.2, random_state=42
)

```


```python
print(X_test.iloc[0]," --- " ,y_test.iloc[0])
```


```python
# Cargamos Stopwords
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = list(set(stopwords.words('spanish')))
```

## Preprocesamiento y Vectorización

Usaremos la técnica TF-IDF (Term Frequency-Inverse Document Frequency):
* TF (Frecuencia de Término): Mide qué tan frecuente es una palabra en un documento.
* IDF (Frecuencia Inversa de Documento): Penaliza las palabras que son muy comunes en todos los documentos (como "el", "la", "un"), dándole más importancia a las palabras que son más distintivas de un texto en particular.





```python


# Crear el vectorizador TF-IDF
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000) # Podemos añadir stop_words en español si queremos

# Aprender el vocabulario y transformar los datos de entrenamiento
X_train_tfidf = vectorizer.fit_transform(X_train)

# Usar el mismo vectorizador para transformar los datos de prueba
X_test_tfidf = vectorizer.transform(X_test)

print("Dimensiones de la matriz TF-IDF de entrenamiento:", X_train_tfidf.shape)
print("Dimensiones de la matriz TF-IDF de prueba:", X_test_tfidf.shape)
```

## Entrenamiento y Evaluación de Modelos
Ahora viene la parte central. Entrenaremos cada uno de nuestros tres modelos con los mismos datos de entrenamiento y los evaluaremos con los mismos datos de prueba.

### Modelo 1: Naive Bayes (Bayesiano Ingenuo)

* Este modelo se basa en el Teorema de Bayes. 
* Calcula la probabilidad de que un texto pertenezca a una clase (ej. "positivo") dadas las palabras que contiene. Se le llama "ingenuo" (naive) porque asume que la presencia de una palabra es independiente de las demás, lo cual no es cierto en el lenguaje.
* Ideal para: Clasificación de texto, es muy rápido y funciona bien con pocas muestras.

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$



Este teorema es fundamental en estadística bayesiana, y se interpreta como: la probabilidad de 
𝐴
dado 
𝐵
es igual a la probabilidad de 
𝐵
dado 
𝐴
, multiplicada por la probabilidad de 
𝐴
, dividida por la probabilidad de 
𝐵
.

| Tarea                          | Por qué funciona bien                                                                                          |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Filtro de spam                 | Las palabras “gratis”, “promoción”, “urgente” tienen distribuciones muy distintas en spam vs. correo legítimo. |
| Análisis de sentimiento básico | Las palabras “excelente” o “horrible” son fuertes pistas de polaridad.                                         |



### Ventajas

* Entrena y predice en milisegundos, incluso con miles de clases.
* Requiere pocos datos para obtener un modelo útil.
* Tolera muy bien vectores dispersos (bolsa-de-palabras, TF-IDF).


### Limitaciones
* La independencia entre palabras rara vez se cumple (p. ej. “no me gustó”).
* No captura interacciones ni pondera contextos complejos.
* Suele dar menor exactitud que modelos discriminativos cuando hay mucho dato etiquetado.




```python
print("--- Entrenando Naive Bayes ---")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Predicciones
y_pred_nb = nb_model.predict(X_test_tfidf)

# Evaluación
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Precisión (Accuracy) de Naive Bayes: {accuracy_nb:.2f}")
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_nb))
```

## Modelo 2: Regresión Logística

* A pesar de su nombre, es un modelo de clasificación. 
* Aprende una función lineal que separa las clases. 
* Luego, usa una función "sigmoide" para "aplastar" el resultado en una probabilidad (un valor entre 0 y 1). 
* * Si la probabilidad es > 0.5, se clasifica como una clase; si no, como la otra.

* Ideal para: Problemas de clasificación binaria donde se busca un modelo interpretable y eficiente. Es un excelente punto de partida (baseline).

Es un modelo discriminativo lineal: aprende pesos 𝑤 para cada característica y estima

$$
P(\text{clase} = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x})
$$

donde 

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$


```python
# --- Regresión Logística ---
print("\n--- Entrenando Regresión Logística ---")
lr_model = LogisticRegression(C=2, solver='liblinear',random_state=42)
lr_model.fit(X_train_tfidf, y_train)

# Predicciones
y_pred_lr = lr_model.predict(X_test_tfidf)

# Evaluación
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Precisión (Accuracy) de Regresión Logística: {accuracy_lr:.2f}")
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_lr))
```

## Modelo 3: Máquina de Soporte Vectorial (SVM)
* Intuición Teórica: La SVM busca encontrar el hiperplano (una línea en 2D, un plano en 3D, etc.) que mejor separa las clases de datos. 
* No solo busca una línea que separe, sino la que tenga el margen más amplio posible entre las clases. Esto lo hace muy robusto contra el ruido.

* Ideal para: Espacios de alta dimensionalidad (como el texto vectorizado) y cuando la separación clara entre clases es posible.


### Ejemplos de casos

* Clasificación de temas de noticias con decenas de miles de palabras: un SVM lineal suele estar en el top-3 de exactitud.
* Detección de insultos en foros: con kernel RBF puede separar patrones de abuso más sutiles que combinan palabras y n-gramas.

#### Ventajas

* Muy eficaz en espacios de alta dimensionalidad y datos dispersos (texto).

* Buena gestión de clases desequilibradas (parámetro class_weight).

* El margen grande tiende a generalizar mejor en tests desconocidos.

#### Limitaciones

* Entrenamiento y predicción más lentos que NB o RL en corpora enormes.

* No produce probabilidades directas (se suele calibrar con Platt scaling).

* Requiere ajustar hiperparámetros (C, kernel), lo que puede ser costoso.


```python

print("\n--- Entrenando SVM ---")
svm_model = SVC(kernel='rbf', random_state=42) 
svm_model.fit(X_train_tfidf, y_train)

# Predicciones
y_pred_svm = svm_model.predict(X_test_tfidf)

# Evaluación
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Precisión (Accuracy) de SVM: {accuracy_svm:.2f}")
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_svm))
```

## Comparación Final de Rendimiento
Ahora que tenemos los resultados de cada modelo, vamos a ponerlos uno al lado del otro para una comparación clara.



```python
# Crear un DataFrame con los resultados para una fácil comparación
results = {
    'Modelo': ['Naive Bayes', 'Regresión Logística', 'SVM'],
    'Accuracy': [accuracy_nb, accuracy_lr, accuracy_svm]
}

results_df = pd.DataFrame(results)

print("\n--- Tabla Comparativa de Rendimiento ---")
print(results_df.sort_values(by='Accuracy', ascending=False))
```

## Primeras Conclusiones

| Criterio                       | Naive Bayes                                               | Regresión Logística                                                                 | SVM                                                                                                      |
| ------------------------------ | --------------------------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Velocidad de entrenamiento** | ★★★★★ (instantáneo)                                       | ★★★★☆                                                                               | ★★☆☆☆ (datasets grandes)                                                                                 |
| **Exactitud típica en texto**  | Buena como baseline                                       | Muy buena                                                                           | Excelente si el tuneo es correcto                                                                        |
| **Probabilidades confiables**  | Aproximadas                                               | SÍ (calibradas)                                                                     | Necesita calibración                                                                                     |
| **Datos necesarios**           | Pocos                                                     | Moderados                                                                           | Muchos (especialmente con kernel)                                                                        |
| **Interpretabilidad**          | Media (palabras con mayor prob.)                          | Alta (pesos directos)                                                               | Media-baja                                                                                               |
| **Casos de uso recomendados**  | Baselines rápidos, streaming, spam, clasificación inicial | Sentiment, intents, riesgo crediticio de textos, cuando se necesitan probabilidades | Categorías finas, grandes corpus de noticias, detección de fraude textual, problemas con margen estrecho |


En nuestro pequeño experimento, podemos observar el rendimiento de cada modelo. Típicamente, para tareas de texto:

* Naive Bayes es un baseline increíblemente rápido y sólido.
* Regresión Logística y SVM (con kernel lineal) suelen competir por el primer puesto e incluso ambos pueden ser más precisos que Naive Bayes.

### Puntos Clave a Recordar:
* No hay un "mejor" modelo universal: El rendimiento depende del dataset, del preprocesamiento y de la tarea específica.
* La importancia del preprocesamiento: La calidad de la vectorización (TF-IDF en este caso) es tan importante como la elección del modelo.
* Fundamentales antes de saltar a arquitecturas más complejas como los Transformers. 




```python
def predict_sentiment(model, vectorizer, text):
    """
    Función para predecir el sentimiento de un texto dado un modelo y un vectorizador.
    """
    text_processed = preprocess_tweet(text)
    text_vectorized = vectorizer.transform([text_processed])
    prediction = model.predict(text_vectorized)
    return prediction[0]

def make_prediction(text, model, vectorizer):
    #norm_text = preprocess_tweet(text)
    sentiment = predict_sentiment(model, vectorizer, text)
    
    sentiment_label = "Positivo" if sentiment == 1 else "Negativo"
    return sentiment_label
```


```python
print(X_test.iloc[2]," --- " ,y_test.iloc[2])

test_case = X_test.iloc[2]
#make_prediction("La gente joven no encuentra sociego", svm_model, vectorizer)
```


```python
test_case
```


```python
make_prediction(test_case, svm_model, vectorizer)
```


```python
make_prediction("Aaaa, pero que rica esta mierda #cocacola", svm_model, vectorizer)
```


```python

```

## Ahora les toca a ustedes: 
* Definan un set de oraciones positivas y negativas para testear cada uno de los modelos.
* Ejecutar la función 


```python
positivas = ["Agrega tu ejemplo"]
negativas = ["Agrega tu ejemplo"]
```


```python
modelos = {
    "Naive Bayes": nb_model,
    "Regresión Logística": lr_model,
    "SVM": svm_model
}

texts = positivas + negativas

resultados = {"Texto": texts}
for nombre, modelo in modelos.items():
    resultados[nombre] = [make_prediction(text, modelo, vectorizer) for text in texts]

comparacion_df = pd.DataFrame(resultados)
comparacion_df
```


```python

```


```python

```


```python
{% include additional_content.html %}
{% include additional_content.html %}
```
