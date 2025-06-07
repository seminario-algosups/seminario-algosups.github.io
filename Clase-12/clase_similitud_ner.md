### Sugerencias de uso de la Notebook: 
- Sugerimos 'Abrir en Colab' y realizar una copia del cuaderno antes de usarlo.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seminario-algosups/seminario-algosups.github.io/blob/master/Clase-12/clase_similitud_ner.ipynb)

# Clase 12: Similitud Semántica y Sistemas de Anotación NER

Esta notebook cubre los siguientes puntos:

**5.c.ii)** Métodos de medición de similitud semántica:  

&nbsp;&nbsp;• Similitud por presencia de tokens (Jaccard)   
&nbsp;&nbsp;• Similitud coseno (TF‑IDF)    
&nbsp;&nbsp;• Word Emebddings  
&nbsp;&nbsp;    • Word2Vec (vectores de palabras)  
&nbsp;&nbsp;    • FastText (sub‑word embeddings)  

**5.b.i)** Sistemas de anotación: BIO / BILOU (con mención a BOU).  
**5.b.ii)** NER con **spaCy** y **Stanza**, y una demo básica de extracción de relaciones.

---


## Definición de un dataset de prueba



```python
import pandas as pd


sentences = [
    "La inteligencia artificial está transformando la industria del software.",
    "La IA revolucionará la asistencia médica en los próximos años.",
    "Los avances en Inteligencia Artificial revolucionaron la detección de patologías en informes médicos.",
    "Los goles de Lionel Messi llevaron al equipo a la victoria.",
    "Las estrategias defensivas del fútbol moderno requieren comunicación constante.",
    "El equipo de fútbol ganó el campeonato después de un partido intenso."
]
df = pd.DataFrame(sentences, columns=["text"])

```

## Métodos de medición de similitud semántica

Veremos cuatro enfoques:

1. **Similitud coseno** entre vectores TF‑IDF.  
2. **Similitud Jaccard** basada en presencia/ausencia de tokens (bolsa de palabras binaria).  
3. **Word2Vec** – calculando la similitud promedio de vectores de palabras (usaremos `es_core_news_md`).  
4. **FastText** – similar al anterior, aprovechando sub‑palabras (opcional si descargas un modelo).


### Indice Jaccard

La **similitud de Jaccard** entre dos conjuntos \(A\) y \(B\) se define como

$$
\text{Jaccard}(A,B) = \frac{|A \cap B|}{|A \cup B|}
$$

ddonde  

* $\displaystyle \left| A \cap B \right|$ — **número de elementos comunes** (intersección)  
* $\displaystyle \left| A \cup B \right|$ — **número total de elementos únicos** presentes en al menos uno de los conjuntos (unión)


Supongamos dos oraciones tokenizadas (sin *stop-words*):

| Oración | Tokens |
|---------|--------|
| **A**: “La inteligencia artificial avanza rápido” | { inteligencia, artificial, avanza, rápido } |
| **B**: “Los avances en inteligencia artificial son impresionantes” | { avances, inteligencia, artificial, impresionantes } |

---
**Intersección**

$$
A \cap B \;=\; \{\textit{inteligencia},\ \textit{artificial}\}
$$

$$
\bigl|A \cap B\bigr| \;=\; 2
$$
**Unión**

$$
A \cup B = \{\textit{inteligencia},\ \textit{artificial},\ \textit{avanza},\ \textit{rápido},\ \textit{avances},\ \textit{impresionantes}\}
$$

$$
\bigl|A \cup B\bigr| = 6
$$

**Cálculo**

$$
J(A,B)= \frac{\bigl|A \cap B\bigr|}{\bigl|A \cup B\bigr|}= \frac{2}{6}\approx 0.33
$$

La similitud de Jaccard indica que las frases comparten aproximadamente un **tercio** de su vocabulario “informativo”.


```python
import numpy as np

def jaccard_similarity(a, b):
    set_a, set_b = set(a.split()), set(b.split())
    inter = set_a & set_b
    union = set_a | set_b
    return len(inter) / len(union)

jaccard_scores = df['text'].apply(lambda x: jaccard_similarity(df['text'][0], x))
pd.DataFrame({'oracion': df['text'], 'sim_jaccard': jaccard_scores}).sort_values('sim_jaccard', ascending=False).head(6)

```

## Similitud de Coseno

La similitud de coseno es una medida de similaridad angular que se utiliza para cuantificar la "cercanía" o "parecido" entre dos vectores no nulos en un espacio de características multidimensional. 

En el contexto de la lingüística computacional, estos vectores suelen representar entidades lingüísticas como palabras, documentos o incluso frases, transformadas numéricamente a través de técnicas como los embeddings o modelos de espacio vectorial de palabras (Word Embedding Models). 

No mide la magnitud de los vectores, sino la orientación de los mismos. 



```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stopwords = ['de', 'la', 'y', 'en', 'a', 'el', 'que', 'los', 'del', 'con', 'por', 'un', 'una', 'me', 'fue', 'tan', 'muy', 'este', 'es', 'para', 'se', 'lo', 'como', 'al', 'si', 'con', 'su', 'misma']

tfidf = TfidfVectorizer(stop_words=stopwords)
X = tfidf.fit_transform(df['text'])

# Similitud coseno respecto a la primera oración
cosine_scores = cosine_similarity(X[0], X).flatten()
pd.DataFrame({'oracion': df['text'], 'sim_coseno': cosine_scores}).sort_values('sim_coseno', ascending=False).head(5)

```

## Word Embeddings
*   Incrustaciones de palabras

*   La idea de que "una palabra se conoce por la compañía que mantiene" (Firth).
    *   Representaciones densas (vectores con muchos números no cero) que capturan el contexto y significado.
    *   Permiten realizar operaciones semánticas: `rey - hombre + mujer ≈ reina`.

*   **Word2Vec (Google):**
    *   Modelo predictivo para aprender embeddings de palabras a partir de un corpus grande.
    *   **Arquitecturas Principales:**
        *   **CBOW (Continuous Bag of Words):** Predice la palabra actual basándose en su contexto.
        *   **Skip-gram:** Predice las palabras del contexto basándose en la palabra actual.
    *   **Similitud:** Una vez entrenado, cada palabra tiene un vector. La similitud entre palabras se calcula típicamente con similitud coseno.
    *   **Ventajas:** Captura relaciones semánticas y sintácticas. Modelos pre-entrenados disponibles.
    *   **Desventajas:** No maneja bien palabras fuera de vocabulario (OOV) si no están en el entrenamiento.


*   **FastText (Facebook):**
    *   Extensión de Word2Vec. Representa cada palabra como una bolsa de n-gramas de caracteres.
    *   **Ejemplo:** La palabra "manzana" con n=3 sería `<ma, man, anz, nza, zan, ana, na>`. El vector de "manzana" es la suma de los vectores de estos n-gramas.
    *   **Ventajas:**
        *   Maneja palabras OOV a partir de la constuccion de un vector a partir de sus n-gramas.
        *   Suele funcionar mejor para lenguajes morfológicamente ricos y para palabras raras.
    *   **Similitud:** Similar a Word2Vec, se usa similitud coseno sobre los vectores de palabras/documentos.

## Uso de vectores propios de Spacy

Revisemos la documentación de spacy en relación a los modelos que usan vectores -- https://spacy.io/models/es#es_core_news_sm:~:text=lemmatizer%2C%20ner-,VECTORS,-500k%20keys%2C%2020k


```python
# !python -m spacy download es_core_news_md
```


```python
import spacy

# Asegurate de tener instalado el modelo con: python -m spacy download es_core_news_md

nlp = spacy.load('es_core_news_md')

def doc_vector(text):
    doc = nlp(text)
    return doc.vector


vecs = df['text'].apply(doc_vector).tolist()
matrix = np.vstack(vecs)
w2v_scores = cosine_similarity([matrix[0]], matrix).flatten()
df_vec = pd.DataFrame({'oracion': df['text'], 'sim_word2vec': w2v_scores}).sort_values('sim_word2vec', ascending=False).head(6)
df_vec
```


```python
print(df_vec['oracion'][0])
print(df_vec['oracion'][1])
```

## Entities y NER

### Entidades

En Procesamiento del Lenguaje Natural hablamos de entidad (o Named Entity, NE) cuando un fragmento de texto ­—una o varias palabras consecutivas— refiere a un objeto concreto del mundo que podemos tipificar: personas, organizaciones, lugares, fechas, cantidades, obras artísticas, etc. 

El reconocimiento de entidades (Named Entity Recognition, NER) consiste en detectar cada mención y asignarle una clase semántica



| Década        | Hitos                                                                                                                                                                                                                                     | Tecnologías dominantes                                    |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **1990-1999** | *Message Understanding Conferences* (MUC-6, 1995) introducen la tarea de NE con tres etiquetas (ENAMEX, TIMEX, NUMEX). Sistemas basados en reglas, diccionarios y expresiones regulares. ([cs.nyu.edu][1], [aclanthology.org][2])         | Gramáticas reguladas, *gazetteers*, expresiones regulares |
| **2000-2009** | *Shared tasks* CoNLL-2002 (español, neerlandés) y CoNLL-2003 (inglés, alemán) fijan corpora estándar y métricas. Aparición de modelos estadísticos discriminativos (Máx. Entropía, CRF). ([aclanthology.org][3], [paperswithcode.com][4]) | HMM, MEMM, CRF                                            |
| **2010-2017** | Primera ola *deep*: vectores distribucionales + BiLSTM-CRF superan a modelos basados en *features* manuales.                                                                                                                              | Word2Vec/GloVe + BiLSTM-CRF                               |
| **2018-hoy**  | Transición a *Transformers*: BERT y derivados alcanzan SOTA con ajuste fino mínimo; surgen modelos multilingües (mBERT) y específicos (BETO, RoBERTa-BNE). ([huggingface.co][5], [machinelearningmastery.com][6])                         | BERT, RoBERTa, GPT, LLMs + *prompting*                    |

[1]: https://cs.nyu.edu/~grishman/muc6.html?utm_source=chatgpt.com "MUC-6"
[2]: https://aclanthology.org/C96-1079.pdf?utm_source=chatgpt.com "[PDF] Message Understanding Conference- 6: A Brief History"
[3]: https://aclanthology.org/W03-0419.pdf?utm_source=chatgpt.com "[PDF] Introduction to the CoNLL-2003 Shared Task - ACL Anthology"
[4]: https://paperswithcode.com/dataset/conll-2003?utm_source=chatgpt.com "CoNLL 2003 Dataset | Papers With Code"
[5]: https://huggingface.co/dslim/bert-base-NER?utm_source=chatgpt.com "dslim/bert-base-NER - Hugging Face"
[6]: https://machinelearningmastery.com/how-to-do-named-entity-recognition-ner-with-a-bert-model/?utm_source=chatgpt.com "How to Do Named Entity Recognition (NER) with a BERT Model"


## Recursos y panorama hispanohablante

El español fue pionero en NER con CoNLL-2002, y hoy dispone de corpora ricos como AnCora-ES (500 000 palabras, múltiples niveles de anotación) muy usado para entrenar y evaluar sistemas de NER, dependencia y coreferencia. [clic.ub.edu]


Modelos pre-entrenados (BETO, bert-base-NER adaptado) ofrecen bases sólidas y pueden ajustarse con bibliotecas como spaCy, Hugging Face o Stanza en pocas líneas de código.

## 3. Sistemas de anotación BIO / BILOU (5.b.i)

En **NLP** se usan esquemas que indican qué tokens pertenecen a entidades nombradas.

| Esquema | Descripción | Ejemplo (`ORG` = *Apple*) |
|---------|-------------|---------------------------|
| **BIO** | **B**egin, **I**nside, **O**utside | Apple = **B-ORG**, Inc. = **I-ORG** |
| **BILOU** | **B**egin, **I**nside, **L**ast, **O**utside, **U**nit (entidad de un solo token) | Apple (**U-ORG**) / University of California = B-ORG I-ORG L-ORG |
| **BOU** | Variante simplificada: **B**egin, **O**utside, **U**nit | (poco usada hoy, incluida por completitud) | Apple (**U-ORG**) |

A continuación generamos etiquetas BIO y BILOU para nuestras oraciones usando *spaCy*.



```python
from spacy.training import offsets_to_biluo_tags


texto = "Apple Inc. lanzó el nuevo iPhone en California."
doc = nlp(texto)

# Obtenén spans de entidades [(start_char, end_char, label), ...]
ents_offsets = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
biluo_tags = offsets_to_biluo_tags(doc, ents_offsets)
bio_tags = [tag.replace('L-', 'I-').replace('U-', 'B-') for tag in biluo_tags]

for token, biluo, bio in zip(doc, biluo_tags, bio_tags):
    print(f"{token.text:<12} {biluo:<8} {bio}")

```

## 4. NER con spaCy + Extracción de relaciones (5.b.ii)

A continuación comparamos los *pipelines* de reconocimiento de entidades.



```python

doc = nlp("Barcelona FC fichó a Lionel Messi en 2003.")
print("spaCy:", [(ent.text, ent.label_) for ent in doc.ents])

def extract_svo(doc):
    svos = []
    for token in doc:
        if token.dep_ == "ROOT":
            subj = [w for w in token.lefts if w.dep_.startswith("nsubj")]
            obj  = [w for w in token.rights if w.dep_.startswith(("dobj", "obj"))]
            if subj and obj:
                svos.append((subj[0].text, token.text, obj[0].text))
    return svos


print("SVO:", extract_svo(doc))

```


```python

def _span_for_ent_token(token):
    """Devuelve el texto completo de la entidad que contiene al token
       o None si el token no pertenece a ninguna entidad."""
    if token.ent_type_ == "":
        return None
    for ent in token.doc.ents:
        if ent.start <= token.i < ent.end:
            return ent.text
    return None

def _expand_to_np(token):
    """Devuelve el texto del sintagma nominal gobernado por el token."""
    subtree = list(token.subtree)
    left   = subtree[0].i
    right  = subtree[-1].i + 1
    return token.doc[left:right].text

def extract_svo_ent(doc):
    svos = []
    for root in doc:
        if root.dep_ == "ROOT":
            subjs = [w for w in root.lefts  if w.dep_.startswith("nsubj")]
            objs  = [w for w in root.rights if w.dep_.startswith(("dobj", "obj"))]
            if subjs and objs:
                def full_phrase(tok):
                    # 1) usa la entidad completa si existe
                    span = _span_for_ent_token(tok)
                    if span:
                        return span
                    # 2) si no, usa el sintagma nominal
                    return _expand_to_np(tok)
                svos.append((full_phrase(subjs[0]),
                              root.lemma_,                # verbo en lema
                              full_phrase(objs[0])))
    return svos


```


```python
print("spaCy:", [(ent.text, ent.label_) for ent in doc.ents])
print("SVO:", extract_svo_ent(doc))
```

{% include additional_content.html %}


