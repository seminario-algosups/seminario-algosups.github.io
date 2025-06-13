### Sugerencias de uso de la Notebook: 
- Sugerimos 'Abrir en Colab' y realizar una copia del cuaderno antes de usarlo.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seminario-algosups/seminario-algosups.github.io/blob/master/Clase-13/clase13.ipynb)

# Analisis de sentimiento

El análisis de sentimiento es una tarea de procesamiento del lenguaje natural que forma parte del amplio campo de las tareas de clasificación de textos (*text classification*).

La clasificación de textos incluye además, entre otras tareas, la detección de spam, la identificación de idioma, la atribución de autoría y la detección de tópicos (*topic detection*). Más recientemente, se incorporó el *toxicity detection*.

El análisis de sentimiento, puntualmente, consiste en determinar automáticamente si un texto resulta positivo, neutral o negativo.

Existen tres grandes enfoques:
- *Enfoques léxicos*: Calculan el sentimiento en función de la presencia o no de ciertas palabras predefinidas de antemano, como, por ejemplo, subjetivemas, negación, predicados de gusto, etc. Están basados en reglas, lo que los hace menos flexibles.
- *Enfoques de aprendizaje automático*: Son mucho más adecuados y flexibles, pero requieren de datos etiquetados.
- *Aprendizaje profundo con transformers preentrenados*: Son el estado del arte actual, pero requieren muchos recursos computacionales.


## Enfoques Léxicos


```python

# Simulamos diccionarios léxicos simples
positivas = {"feliz", "bueno", "excelente", "maravilloso", "fantástico"}
negativas = {"malo", "horrible", "terrible", "feo", "asco"}

def analisis_lexico(texto):
    tokens = texto.lower().split()
    pos = sum(1 for t in tokens if t in positivas)
    neg = sum(1 for t in tokens if t in negativas)
    if pos > neg:
        return "POSITIVO"
    elif neg > pos:
        return "NEGATIVO"
    else:
        return "NEUTRO"

# Ejemplo
analisis_lexico("El día fue maravilloso pero la comida estuvo horrible")
```

## Enfoques de ML


```python
from datasets import load_dataset

# Cargar el dataset desde Hugging Face
dataset = load_dataset("mrm8488/tass-2019")
```


```python
dataset["train"][0]  # Ver el primer ejemplo del dataset
print(len(dataset["train"]))  # Ver el número de ejemplos en el dataset
print(len(dataset["test"]))
```


```python
from pprint import pprint
pprint(dataset["train"][:10])
```


```python
import pandas as pd

#sampled = dataset["train"].train_test_split(test_size=0.5, seed=42)["test"]
#print(f"Número de ejemplos muestreados: {len(sampled)}")
#print(sampled[0])

# Convertir a DataFrame
df = pd.DataFrame(dataset["train"])
df = df[["sentence", "labels"]].dropna()

# Ver etiquetas únicas
df["labels"].value_counts()
```


```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["sentence"])
y = df["labels"]

```


```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
```


```python
textos = [
    "Me encanta este producto", "Odio este lugar",
    "La película fue buena", "Es horrible y lento",
    "Excelente atención", "Malísimo todo", "La comida estuvo bien, la atención podria mejorar bastante"
]
# Probar con el ds original!!
etiquetas = ["pos", "neg", "pos", "neg", "pos", "neg", "neg"]
# 0: neg, # 1: pos
df_testeos = pd.DataFrame({"text": textos, "label": etiquetas})

df_testeos["pred"] = clf.predict(vectorizer.transform(df_testeos["text"]))
```


```python
df_testeos
```

## Modelos basados en Transformers: 
1. Verificado
2. PySentimiento




```python
# Usando un modelo preentrenado de Hugging Face para análisis de sentimientos
from transformers import pipeline

pipe = pipeline("text-classification", model="VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis")

for texto in textos:
    resultado = pipe(texto)
    print(f"Texto: {texto} -> Etiqueta: {resultado[0]['label']}, Score: {resultado[0]['score']:.4f}")
```

## PySentimiento: Análisis de Opinión y NLP Social Multilingüe

`pysentimiento` es una biblioteca de Python que ofrece modelos preentrenados para tareas de procesamiento de lenguaje natural centradas en redes sociales. Es compatible con varios idiomas (español, inglés, portugués, italiano) y permite realizar análisis de sentimiento, detección de emociones, discurso de odio, ironía, análisis de entidades nombradas (NER), y etiquetas gramaticales (POS).

A continuación, se muestran ejemplos de uso con texto en español.



```python

# Instalar pysentimiento (si no está instalado)
!pip install pysentimiento

```

## Flujo: del texto a la predicción

Para entender cómo funciona internamente la herramienta, aquí se describe el pipeline típico que ocurre al llamar `analyzer.predict(text)`:

### 1. Preprocesamiento
El texto crudo se pasa primero por el preprocesador de tweets (a menos que esté deshabilitado). Este paso limpia el texto aplicando reglas como la normalización de menciones y URLs, la reducción de repeticiones de caracteres, la conversión de emojis a texto, etc. El resultado es una versión normalizada del texto de entrada.

### 2. Tokenización
El tokenizador del analizador (una subclase de `AutoTokenizer` de Hugging Face) convierte el texto preprocesado en una secuencia de tokens/IDs que el modelo Transformer puede interpretar. Esto incluye dividir el texto en subpalabras y agregar tokens especiales (como `[CLS]` y `[SEP]` para BERT, o solo tokens de inicio/fin en el caso de RoBERTa). También se realiza padding o truncamiento según la longitud máxima permitida.

### 3. Inferencia del Modelo
Los IDs de tokens se envían al modelo Transformer (que internamente es un modelo de PyTorch). Dependiendo de la tarea:

- **Tareas de clasificación:** El modelo produce un conjunto de logits (puntuaciones no normalizadas) para cada clase. Por ejemplo, el modelo de sentimiento devuelve tres logits (uno por cada clase: POS, NEG, NEU).
- **Tareas multi-etiqueta:** El modelo devuelve un logit por cada etiqueta posible (por ejemplo, tres logits para las etiquetas de discurso de odio: `hateful`, `targeted`, `aggressive`).
- **Tareas de etiquetado secuencial (NER/POS):** Se devuelve un logit por token por cada posible etiqueta.

### 4. Post-procesamiento de la Predicción
Los logits producidos por el modelo se convierten en probabilidades:

- Para clasificación de una sola etiqueta, se aplica **softmax**, y se selecciona la etiqueta con mayor probabilidad como `output`.
- Para clasificación multi-etiqueta (como discurso de odio), se aplica **sigmoid** a cada logit, y se seleccionan las etiquetas cuya probabilidad supera un umbral (por defecto, 0.5).


Luego, las etiquetas seleccionadas se ensamblan en un objeto 'resultado'.

### 5. Formato de Salida
El resultado se devuelve como un objeto `AnalyzerOutput` (o una lista si se procesan múltiples textos). Este objeto contiene:

- `output`: la etiqueta predicha (para tareas de una sola etiqueta) o una lista de etiquetas (para tareas multi-etiqueta). En tareas secuenciales como NER o POS, puede ser una lista de anotaciones por token.
- `probas`: un diccionario que asigna a cada etiqueta su probabilidad estimada.  
  Por ejemplo:
  ```python
  {'POS': 0.998, 'NEG': 0.002, 'NEU': 0.000}
  ```
  para una oración claramente positiva, o:
  ```python
  {'hateful': 0.987, 'targeted': 0.978, 'aggressive': 0.969}
  ```
  ppara un texto con discurso de odio.

## Veamos algunos Ejemplos de funcionalidades

Tras la instalación, el uso de la biblioteca es muy sencillo. 

Se crea un analizador para la tarea y el idioma de interés y, a continuación, se llama a la funcion 'predict' con nuevos textos. 

En este caso definimos como tarea el análisis de sentimiento y el idioma que queremos usar como español



```python

from pysentimiento import create_analyzer

# Analizador de sentimiento en español
sentiment_analyzer = create_analyzer(task="sentiment", lang="es")

```


```python

texto_positivo = "Qué gran jugador es Messi"
texto_negativo = "Esto es pésimo"
texto_neutro = "El evento será a las 5 PM"

print("Positivo:", sentiment_analyzer.predict(texto_positivo))
print("Negativo:", sentiment_analyzer.predict(texto_negativo))
print("Neutro:", sentiment_analyzer.predict(texto_neutro))

```

## Emotion Analysis

`pysentimiento` provee análisis de emociones a través de modelos pre-entrenados con los datasets de [EmoEvent](https://github.com/fmplaza/EmoEvent-multilingual-corpus/)


```python

emotion_analyzer = create_analyzer(task="emotion", lang="es")

print("Emoción:", emotion_analyzer.predict("Estoy muy feliz con el resultado"))

```

## Hate Speech
Replicamos la operación pero esta ver para análissi de odio


```python

hate_analyzer = create_analyzer(task="hate_speech", lang="es")

texto_odio = "Vaya guarra barata y de poca monta es XXXX!"
print("Discurso de odio:", hate_analyzer.predict(texto_odio))

```


```python

irony_analyzer = create_analyzer(task="irony", lang="es")

texto_ironia = "Sí claro, como si el gobierno alguna vez resolviera algo..."
print("Ironía:", irony_analyzer.predict(texto_ironia))

```

## Token Labeling tasks


`pysentimiento` cuenta con analizadores para POS tagging & NER gracias al dataset multilingual [LinCE](https://ritual.uh.edu/lince/)


```python

from pysentimiento.preprocessing import preprocess_tweet

texto_tweet = "@usuario deberías ver esto http://bit.ly/ejemplo jajajajajaja"
print("Texto original:", texto_tweet)
print("Preprocesado:", preprocess_tweet(texto_tweet))

```


```python

ner_analyzer = create_analyzer(task="ner", lang="es")

print("Entidades nombradas:", ner_analyzer.predict("Lionel Messi juega en el Inter de Miami"))

```


```python

pos_analyzer = create_analyzer(task="pos", lang="es")

print("POS tagging:", pos_analyzer.predict("Messi corre rápidamente con el balón"))

```


## 🧠 Ejercicio: Análisis de Sentimientos y Emociones en Opiniones de Usuarios

### 📝 Objetivo
Usar `pysentimiento` para analizar una serie de opiniones extraídas de redes sociales o reseñas, y comparar los resultados obtenidos con la intuición humana. Reflexionar sobre aciertos, errores y posibles fuentes de ambigüedad.

### 📦 Materiales
- Lista de 10 a 15 frases que simulen opiniones de usuarios (pueden ser reales o inventadas).
- Un cuaderno (notebook) en Colab o Jupyter con `pysentimiento` instalado.

### 🔧 Instrucciones

1. **Preparación del Dataset**





```python
opiniones = [
       "Me encantó la atención al cliente, volvería sin dudarlo.",
       "Una pérdida de tiempo, el producto no sirve.",
       "El hotel estaba bien, aunque el desayuno era mejorable.",
       "No sé qué pensar, fue una experiencia rara.",
       "¡Excelente servicio y muy rápido!",
       "Demasiado caro para lo que ofrece.",
       "No me gustó, pero entiendo que a otros sí.",
       "La película fue una joya, me hizo llorar de emoción.",
       "Malisimo, no lo recomiendo a nadie.",
       "La app está bien diseñada, pero falla mucho."
   ]

```

## 2. Predicción automática con PySentimiento



```python
from pysentimiento import create_analyzer

sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
emotion_analyzer = create_analyzer(task="emotion", lang="es")

for frase in opiniones:
    sentimiento = sentiment_analyzer.predict(frase)
    emocion = emotion_analyzer.predict(frase)
    print(f"Frase: {frase}")
    print(f"  ➤ Sentimiento: {sentimiento.output}, probabilidades: {sentimiento.probas}")
    print(f"  ➤ Emoción: {emocion.output}, probabilidades: {emocion.probas}")
    print("-" * 80)

```

### Tareas a realizar

🧐 Clasificar cada frase como positiva, negativa o neutra antes de ejecutar el modelo. Luego comparalo con la predicción automática.

❓ ¿Coincide tu intuición con la del modelo? ¿En qué casos no? ¿Por qué podría fallar el modelo?

💡 Observar los casos donde el modelo predice neutral. ¿Te parece apropiado? ¿Qué tipo de lenguaje aparece allí?

🔍 Revisar también las emociones detectadas. ¿Hay ambigüedad? ¿Captura bien la emoción dominante?

🧪 Modificar algunas frases para ver cómo cambian las predicciones. Por ejemplo: "El servicio fue increíblemente lento" vs. "El servicio fue lento, pero aceptable".

## Reflexión Final

¿Qué fortalezas y limitaciones encontraste en el modelo?

¿Qué tipos de sesgos podrían surgir si solo usamos este tipo de modelos para tomar decisiones?

¿Cómo mejorarías este pipeline para un caso de uso real, como popr ej. para  monitorear opiniones sobre una marca?

   
Para más información: [https://github.com/pysentimiento/pysentimiento](https://github.com/pysentimiento/pysentimiento)

