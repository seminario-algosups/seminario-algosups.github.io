# Clase - 07 - Procesamiento morfológico - (Virtual - Sábado 3 de Mayo)



```python

```

### Sugerencias de uso de la Notebook: 
- Sugerimos 'Abrir en Colab' y realizar una copia del cuaderno antes de usarlo.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seminario-algosups/seminario-algosups.github.io/blob/master/Clase-07/clase-07.ipynb)


*   **Conceptos Clave:**
    *   **Raíz (Root):** El núcleo léxico irreductible de una palabra una vez que se han eliminado todos los afijos flexivos y derivativos. *Ejemplo: 'port-' en 'portador', 'exportar', 'transporte'.*
    *   **Lema (Lemma):** La forma *canónica* o de diccionario de una palabra. Representa el conjunto de todas sus formas flexionadas. *Ejemplo: 'correr' es el lema de 'corro', 'corres', 'corrió', 'corriendo'.*
    *   **Tema (Stem):** La parte de la palabra a la que se añaden los afijos flexivos. Puede incluir la raíz y afijos derivativos. *Ejemplo: En español, para verbos, a menudo coincide con la raíz + vocal temática (e.g., 'cant-a-' en 'cantamos'). En inglés, 'running' -> stem 'runn' (con duplicación), 'cats' -> stem 'cat'. (Nota: La definición precisa varía entre teorías y lenguas)*.
    *   **Forma de Palabra | Forma léxica (Word Form):** La palabra tal como aparece en el texto. *Ejemplo: 'corrieron', 'gatos', 'beautifully'.*
*   **Tipos de Morfología:**
    *   **Morfología Flexiva (Inflectional Morphology):** Modifica una palabra para expresar diferentes categorías gramaticales (tiempo, número, género, caso, etc.) sin cambiar su categoría léxica central. *Ejemplo: 'cantar' -> 'cantó' (tiempo), 'gato' -> 'gatos' (número).* El resultado es una *forma de palabra* diferente del mismo *lema*.
    *   **Morfología Derivativa (Derivational Morphology):** Crea nuevas palabras (nuevos lemas) a partir de otras, a menudo cambiando la categoría léxica. *Ejemplo: 'nación' (sustantivo) -> 'nacional' (adjetivo) -> 'nacionalizar' (verbo).*

En cualquier tarea de NLP (clasificación, similitud, ebeddings, etc.) resulta importante agrupar o normalizar palabras y estas nociones nos ayudan en ese objeticvo.

## Procesamiento Morfológico Básico con NLTK

[NLTK](https://www.nltk.org/#natural-language-toolkit)  *is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum.*


```python
# (Pre-requisito) Instalación y Descarga de Recursos NLTK

#!pip install nltk  # Descomentar y ejecutar si nltk no está instalado en el entorno
```


```python

import nltk

# Descargar recursos necesarios (solo la primera vez que se ejecuta en un entorno)
print("Descargando recursos de NLTK...")
nltk.download('punkt', quiet=True) # Para tokenización
nltk.download('punkt_tab', quiet=True) # Español
print("Punkt descargado.")
nltk.download('wordnet', quiet=True) # Para lematizador WordNet
print("WordNet descargado.")
nltk.download('omw-1.4', quiet=True) # Open Multilingual Wordnet
print("OMW 1.4 descargado.")
print("Recursos listos.")
```

*   **Tokenización:**
    *   **Concepto:** Dividir el texto en unidades significativas (palabras, puntuación), llamadas *tokens*. Es el primer paso en la mayoría de los pipelines de NLP.
    *   **Implementación NLTK:**


```python
from nltk.tokenize import word_tokenize

texto = "Las ideas verdes incoloras duermen relocas."
tokens = word_tokenize(texto, language='spanish')
print(tokens)

```

*   **Stemming:**
    *   Proceso para reducir las palabras flexionadas a una forma base común o "raíz" (stem).
    *   A menudo, crudo y basado en reglas o heurísticas.
    *   Intenta aproximarse a la raíz o al tema mediante el recorte de los tokens a partir de sufijos comunes.
    *     No siempre resulta en una palabra real o en el lema lingüístico correcto.
    *     Es rápido pero menos preciso que la lematización.

**Implementación NLTK**


```python
from nltk.stem.snowball import SnowballStemmer

stemmer_es = SnowballStemmer('spanish')
# tokens = ['corriendo', 'corredor', 'correrán', 'casas', 'casero'] # Chequear que  'corredor' y 'casero' (derivación) pueden o no reducirse dependiendo del stemmer.

stems = [stemmer_es.stem(t) for t in tokens]
print(stems)

#
```

* Ventajas: velocidad, simplicidad
* Desventajas: sobrerreducción, no siempre produce palabras reales, ignora el contexto

## **Lemmatizing (Lemmatization):**
*   Proceso más sofisticado que el stemming, que busca reducir la palabra a su *lema* (forma de diccionario)
*   Tiene en consideración el contexto gramatical (categoría gramatical o Part-of-Speech - POS).
*   Intenta encontrar el lema lingüístico correcto dado que maneja de forma acertada la flexión y algunos casos de derivación.

    
### Implementación NLTK (usando WordNet):
    



```python
from nltk.stem import WordNetLemmatizer
# El POStag es crucial para el lematizadoer (v=verbo, n=nombre, a=adjetivo, r=adverbio)

lemmatizer = WordNetLemmatizer()

print("Lematización 'running' (verbo):", lemmatizer.lemmatize('running', pos='v')) #
print("Lematización 'ran' (verbo):", lemmatizer.lemmatize('ran', pos='v'))
print("Lematización 'cats' (nombre):", lemmatizer.lemmatize('cats', pos='n'))
print("Lematización  'better' (adjetivo):", lemmatizer.lemmatize('better', pos='a'))

# Ejemplo sin POS tag (lo asume como  nombre por defecto):
print("Lematización  'running' (sin POS):", lemmatizer.lemmatize('running'))
```

Ventajas: produce palabras reales/lemas, más preciso lingüísticamente.
Desventajas: más lento, requiere POS tagging para buena precisión, dependencia de lexicones

## Lematización para español con spaCy

[spaCy](https://spacy.io)  librería moderna y eficiente para NLP, orientada a la producción, con modelos pre-entrenados para varios idiomas.


```python
# (Pre-requisito) Instalación de spaCy y Descarga de Modelos

# !pip install -q spacy # Instañlá spacy si no está presente (el -q es para modo silencioso)
# !python -m spacy download es_core_news_lg # Descarga modelo grande para español (usa _md o _sm para medianos/pequeños)

```


```python

import spacy

# Cargar el modelo de español
# Considerá usar modelos más pequeños ('es_core_news_sm', 'es_core_news_md') si los recursos son limitados
# Los modelos más grandes ('lg') suelen tener mejor rendimiento en lematización y otras tareas.
nlp = spacy.load('es_core_news_lg')

# Si necesitaras inglés:
# !python -m spacy download en_core_web_sm
# nlp_en = spacy.load('en_core_web_sm')
```

*   **Procesamiento con spaCy:**
    *   spaCy realiza múltiples tareas (tokenización, POS tagging, análisis de dependencias, NER, *lematización*) en un solo paso al procesar el texto con el objeto `nlp`.


```python
# texto = "Los corredores corrían rápidamente hacia las casas."
doc = nlp(texto)

# Iterar sobre los tokens y acceder a sus atributos
print(f"{'Token:':<15} {'Lema:':<15} {'POS:'}")
for token in doc:
    print(f"{token.text:<15} {token.lemma_:<15} {token.pos_}")


```

### Comparacion Stem & Lemma


```python
frase_ejercicio = texto # "Las investigadoras desarrollaron nuevas metodologías analíticas."

# 2. Tokenizar con NLTK
tokens_nltk = word_tokenize(frase_ejercicio, language='spanish')

# 3. Stemming con NLTK
stemmer_ej = SnowballStemmer('spanish')
stems_nltk = [stemmer_ej.stem(token) for token in tokens_nltk]

# 4. Procesar con spaCy para lematización
doc_spacy = nlp(frase_ejercicio)

# Asegurarse de que la tokenización de spaCy coincida o manejar desajustes
# spaCy puede tokenizar de forma ligeramente diferente (ej. puntuación)
tokens_spacy = [token.text for token in doc_spacy]
lemas_spacy = [token.lemma_ for token in doc_spacy]

# 5. Imprimir comparación (Alineando resultados si la tokenización difiere)
# Una forma simple es iterar sobre los tokens de spaCy ya que incluye puntuación
print(f"{'Token (spaCy)':<15} {'Stem (NLTK)':<15} {'Lema (spaCy)':<15}")
print("-"*45)

# Creamos un diccionario de stems NLTK para búsqueda rápida
# OJO: Esto asume que word_tokenize y la tokenización de spaCy producen tokens similares
# Puede requerir una alineación más compleja en casos difíciles.
stem_dict = {token: stem for token, stem in zip(tokens_nltk, stems_nltk)}

for token in doc_spacy:
    # Intentar encontrar el stem correspondiente del token NLTK
    # Se usa token.text.lower() para buscar por si acaso hay diferencias de mayúsculas/minúsculas
    # y se provee un default si el token de spaCy (ej. '.') no está en los de NLTK
    stem = stem_dict.get(token.text, 'N/A')
    if stem == 'N/A' and token.text.lower() in stem_dict:
       stem = stem_dict[token.text.lower()]

    print(f"{token.text:<15} {stem:<15} {token.lemma_:<15}")

```


```python

```


```python

```

## **Lematización en spaCy: ¿Reglas o Aprendizaje Automático?**
spaCy combina diferentes estrategias para la lematización dependiendo del idioma y del modelo.

*   **Lematizador Basado en Reglas (Rule-based Lemmatizer):**
    *   Utiliza tablas de búsqueda (léxicos) y reglas morfológicas definidas explícitamente para mapear formas flexionadas a lemas.
    *   *Ejemplo:* Una regla podría decir "si una palabra termina en '-ando' o '-iendo' y su POS es VERB, quitar la terminación y añadir 'r' (ajustando la vocal temática si es necesario)". Una tabla podría listar excepciones como 'soy' -> 'ser'.
    *   spaCy permite añadir o modificar estas reglas. Generalmente se usa para excepciones o casos específicos del idioma.
*   **Lematizador Basado en Aprendizaje Automático (Machine Learning-based Lemmatizer):**
    *   Muchos modelos de spaCy (especialmente los más grandes y recientes como `es_core_news_lg`) utilizan componentes entrenados con aprendizaje automático.
    *   Estos modelos aprenden patrones a partir de grandes cantidades de datos anotados (como corpus con lemas correctos) y pueden predecir el lema basándose en la forma de la palabra, su contexto y su POS tag (que también suele predecir un modelo).
    *   Suelen ser más robustos ante palabras desconocidas o variaciones no cubiertas por las reglas, pero pueden cometer errores inesperados.
    
*   **¿Cómo sabe spaCy cuál usar?** La configuración del pipeline del modelo cargado (`nlp.pipe_names`, `nlp.analyze_pipes()`) indica qué componentes están activos. Algunos modelos pueden usar principalmente reglas (a menudo almacenadas en `spacy-lookups-data`), otros pueden usar un componente entrenado (`tagger`, `lemmatizer` que dependen de `tok2vec`), y a menudo es una combinación. Los modelos más grandes (`_md`, `_lg`) tienden a apoyarse más en componentes entrenados.


```python
# Ver los componentes del pipeline actual
print(f"Componentes en el pipeline '{nlp.meta['name']}': {nlp.pipe_names}")

# Análisis más detallado (muestra qué componente asigna qué atributo, como token.lemma)
print("\nAnálisis detallado del pipeline:")
print(nlp.analyze_pipes(pretty=True))
```



## Etiquetamiento en Formato CoNLL-U

*   **CoNLL-U:**
    *   **¿Qué es?** Un formato de texto plano, basado en columnas separadas por tabuladores, utilizado para anotar corpus lingüísticos. Es un estándar de facto en el proyecto Universal Dependencies y ampliamente usado en NLP.

    *   **Estructura:** Una línea por token (o palabra multi-token), columnas fijas con información específica. Comentarios empiezan con `#`. Frases separadas por líneas en blanco.

*   **Columnas Relevantes:**
    *   **Columna 1: ID:** Índice del token en la frase. Suele ser numérico (1, 2, 3...). Puede indicar rangos para tokens multi-palabra (e.g., 'del' -> 'de el') o números decimales para palabras "vacías" insertadas en análisis elípticos.
    *   **Columna 2: FORM:** La forma de la palabra tal como aparece en el texto original. Corresponde a la *forma de palabra* discutida en morfología.
    *   **Columna 3: LEMMA:** El lema o forma base/diccionario de la palabra. Corresponde al *lema*  o  al resultado esperado de la lematización.
    *   **Otras columnas importantes** UPOSTAG (POS universal), XPOSTAG (POS específico del idioma), FEATS (rasgos morfológicos), HEAD (ID del token del que depende), DEPREL (relación de dependencia), DEPS (dependencias secundarias), MISC (anotaciones misceláneas).
*   **Ejemplo:**
    ```conllu
    # sent_id = 1
    # text = Los corredores corrían.
    1   Los         el          DET    Definite=Def|Gender=Masc|Number=Plur|PronType=Art   2   det     _   _
    2   corredores  corredor    NOUN   Gender=Masc|Number=Plur                            3   nsubj   _   _
    3   corrían     correr      VERB   Mood=Ind|Number=Plur|Person=3|Tense=Imp|VerbForm=Fin 0   root    _   _
    4   .           .           PUNCT  _                                                  3   punct   _   _
    ```
*   **Análisis del Ejemplo:**
    *   En la línea 1: ID=1, FORM="Los", LEMMA="el".
    *   En la línea 2: ID=2, FORM="corredores", LEMMA="corredor".
    *   En la línea 3: ID=3, FORM="corrían", LEMMA="correr".
    *   En la línea 4: ID=4, FORM=".", LEMMA=".".
*   **Relevancia:** Este formato permite almacenar y compartir de manera estandarizada los resultados de análisis como la tokenización (FORM) y la lematización (LEMMA), junto con otra información lingüística valiosa.






```python

# Instalar la librería si no está
# !pip install spacy-conll

```


```python


import spacy
from spacy_conll.formatter import ConllFormatter


# --- Añadir el formateador CoNLL-U al pipeline ----
# Es importante añadirlo DESPUÉS de los componentes que generan la info necesaria
# (parser para dependencias, morphologizer/tagger para POS/feats, lemmatizer)
# Si el modelo ya tiene estos componentes, podemos añadirlo al final.

if "conll_formatter" not in nlp.pipe_names:
     # Usar las columnas estándar de CoNLL-U
    config = {"conversion_maps": {"dep": {"nsubj": "nsubj"}}} # Ejemplo mínimo, a chequear si no es necesario
    formatter = nlp.add_pipe("conll_formatter", config=config, last=True)

texto_es = texto # "El rápido zorro marrón salta sobre el perro perezoso."
doc = nlp(texto_es)

# Acceder al formato CoNLL-U a través de la extensión ._.conllu_str
conllu_output = doc._.conll_str
print(conllu_output)

```


```python
conllu_output_df = doc._.conll_pd # Lo convertimos en un DF de pandas para facilitar la manipulación y visualización.
print(conllu_output_df[['ID', 'FORM', 'LEMMA', 'UPOS']])
```


```python

```

## Parte 2: TP - 1
* Objetivo: Analizar cómo diferentes medios presentan la misma información usando herramientas de NLP y el formato CoNLL-U.


### Paso 1: Limpieza del Texto:
Los textos de noticias online contienen HTML, anuncios, menús, etc. Necesitamos extraer el cuerpo principal de la noticia.

* Método Simple:
 *  Copiar y pegar el texto relevante de cada noticia en archivos de texto plano (.txt) separados (noticia1.txt, noticia2.txt, noticia3.txt).
 *  Eliminar manualmente encabezados, pies de página, publicidad  u otro tipo de ruido.
 *  Guardar los archivos respetando los nombres y cargarlos en la notebook. Podés usar el menu de la izquierda.

### Generación de Anotaciones CoNLL-U con spaCy:



```python
import re
import string
from pathlib import Path

def preprocess_spanish_text(file_path: str) -> str:
    """
    Lee un archivo de texto en español, elimina líneas en blanco y aplica
    preprocesamiento básico de NLP.

    Pasos de preprocesamiento:
    1.  Lee el archivo línea por línea.
    2.  Elimina líneas que estén completamente en blanco o contengan solo espacios.
    3.  Une las líneas restantes en un solo bloque de texto.
    4.  Convierte todo el texto a minúsculas.
    5.  Elimina signos de puntuación (reemplazándolos con espacios para evitar unir palabras).
    6.  Elimina números (reemplazándolos con espacios).
    7.  Elimina espacios en blanco extra (múltiples espacios, tabulaciones, etc.),
        dejando solo un espacio entre palabras.
    8.  Elimina espacios en blanco al principio y al final del texto resultante.

    Args:
        file_path (str): La ruta al archivo .txt que se va a procesar.

    Returns:
        str: El texto preprocesado como una única cadena de texto.

    Raises:
        FileNotFoundError: Si el archivo especificado en file_path no existe.
        IOError: Si ocurre un error al leer el archivo.
    """
    input_path = Path(file_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"El archivo no fue encontrado en: {file_path}")

    try:
        #
        with input_path.open('r', encoding='utf-8') as f:
            lines = f.readlines()
    except IOError as e:
        raise IOError(f"Error al leer el archivo {file_path}: {e}")

    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip() # Elimina espacios/saltos al inicio/final de la línea
        if stripped_line: # Si la línea no está vacía después de quitar espacios
            cleaned_lines.append(stripped_line)

    # Unir con un espacio para asegurar separación entre palabras de líneas adyacentes
    full_text = ' '.join(cleaned_lines)
    text = full_text.lower()



    # Definir puntuación a eliminar!
    # Podríamos querer agregar mas  símbolos si es necesario: '–', '—', ... string.punctuations puede ir tmb
    punctuation_to_remove = string.punctuation + '¡¿«»“”‘’!?'
    # Crear una tabla de traducción: cada carácter de puntuación se mapea a un espacio
    translator = str.maketrans(punctuation_to_remove, ' ' * len(punctuation_to_remove))
    text = text.translate(translator)
    # Reemplazar secuencias de dígitos con un espacio
    text = re.sub(r'\d+', '00', text)
    # print(text)
    # Reemplazar una o más ocurrencias de espacios/tabs/newlines con un solo espacio
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text
```


```python
# Asegurarse de tener spacy, es_core_news_lg y spacy-conllu cargados en el entorno. Si no, ejecutar más arriba.)
# Asegurarse tmb de que los archivos se llamen noticia1.txt y estésn cargados en el entorno.


from pathlib import Path


nombres_archivos_txt = ["noticia1.txt", "noticia2.txt"]
nombres_archivos_conllu = ["noticia1.conllu", "noticia2.conllu"]

print("Procesando archivos y generando CoNLL-U...")

for txt_file, conllu_file in zip(nombres_archivos_txt, nombres_archivos_conllu):
    print(f"  Procesando {txt_file}...")
    try:
        ruta_txt = Path(txt_file)
        # texto_noticia = ruta_txt.read_text(encoding="utf-8")
        text_norm = preprocess_spanish_text(ruta_txt) # Lo mandamos a preprocesar
        doc = nlp(text_norm)

        conllu_output = doc._.conll_str # me quedo con el resultado de conllu
        ruta_conllu = Path(conllu_file)
        ruta_conllu.write_text(conllu_output, encoding="utf-8") # lo guardo
        print(f"    -> Guardado en {conllu_file}")

    except FileNotFoundError:
        print(f"    ERROR: Archivo {txt_file} no encontrado. Asegúrate de que exista.")
    except Exception as e:
        print(f"    ERROR procesando {txt_file}: {e}")

print("Proceso completado.")
```


```python
#!pip install conllu
```


```python
from conllu import parse_incr
from collections import Counter


def analizar_conllu(filepath):
    """Lee un archivo CoNLL-U y extrae algunas estadísticas."""
    tokens_data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            tokens_data.extend(tokenlist) # Acumula tokens de todas las frases

    num_tokens = len(tokens_data)
    num_sentences = len(set(t['id'][0] for t in tokens_data if isinstance(t['id'], tuple))) # Aproximación contando sent_id si existe, o reinicios de ID=1
    if not num_sentences: # Fallback si no hay metadatos o IDs son continuos
         num_sentences = sum(1 for t in tokens_data if t['id'] == 1)


    lemas = [t['lemma'].lower() for t in tokens_data if t['lemma'] not in stop_words_es]
    pos_tags = [t['upos'] for t in tokens_data if t['lemma'] not in stop_words_es]
    dep_rels = [t['deprel'] for t in tokens_data if t['lemma'] not in stop_words_es]

    # Contar los 10 lemas más comunes (excluyendo puntuación)
    lemas_comunes = Counter(l for l, p in zip(lemas, pos_tags) if p != 'PUNCT').most_common(10)
    # Contar etiquetas POS
    pos_counts = Counter(pos_tags)
    # Contar relaciones de dependencia
    dep_counts = Counter(dep_rels)

    print(f"\n--- Análisis de: {filepath} ---")
    print(f"  Frases (aprox): {num_sentences}")
    print(f"  Tokens: {num_tokens}")
    if num_sentences > 0:
         print(f"  Tokens/Frase (promedio): {num_tokens/num_sentences:.2f}")
    print(f"  Lemas más comunes: {lemas_comunes}")
    print(f"  Distribución POS: {pos_counts}")
    # print(f"  Distribución DepRel: {dep_counts}") # Puede ser muy largo

    # Podríamos devolver un diccionario o DataFrame para comparar fácilmente
    return {
        "Archivo": filepath,
        "Frases": num_sentences,
        "Tokens": num_tokens,
        "Lemas_Top10": lemas_comunes,
        "POS_Counts": pos_counts
    }

```


```python
# --- Analizar los archivos generados ---
resultados = []
for conllu_file in nombres_archivos_conllu:
     try:
         resultados.append(analizar_conllu(conllu_file))
     except FileNotFoundError:
         print(f"Archivo {conllu_file} no encontrado para análisis.")
     except Exception as e:
         print(f"Error analizando {conllu_file}: {e}")

#  Mostrar comparación en una tabla con Pandas
if resultados:
    df = pd.DataFrame(resultados)

    # print(df)
    for res in resultados:
         print(f"\nResumen para {res['Archivo']}:")
         print(f"  Frases: {res['Frases']}, Tokens: {res['Tokens']}")
         print(f"  Lemas Top 10: {res['Lemas_Top10']}")
         print(f"  POS Counts (Top 5): {Counter(res['POS_Counts']).most_common(5)}")
```


```python

def extract_lemmas_from_conllu(filepath: str) -> list[str]:
    """Lee un archivo CoNLL-U y devuelve una lista de lemas (lowercase, no puntuación)."""
    lemmas = []
    ruta_archivo = Path(filepath)
    if not ruta_archivo.is_file():
        print(f"Advertencia: Archivo no encontrado {filepath}")
        return []
    try:
        with ruta_archivo.open("r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                for token in tokenlist:
                    # Usar .get si falta la columna
                    lemma = token.get('lemma')
                    upos = token.get('upos')
                    # Añadir si el lema existe y no es puntuación, si no se saltea ese lema
                    if lemma and upos != 'PUNCT':
                        lemmas.append(lemma.lower())
    except Exception as e:
        print(f"Error procesando {filepath}: {e}")
        return []
    return lemmas

# Asumiendo que los archivos .conllu se llaman como antes!!
nombres_archivos_conllu = ["noticia1.conllu", "noticia2.conllu"]
lista_de_lemmas_por_noticia = []
corpus_para_tfidf = []

print("Extrayendo lemas de archivos CONLLU...")
for conllu_file in nombres_archivos_conllu:
    lemmas_noticia = extract_lemmas_from_conllu(conllu_file)
    if lemmas_noticia:
        lista_de_lemmas_por_noticia.append(lemmas_noticia)
        # Unir los lemas en un string para tfidfVectorizer
        corpus_para_tfidf.append(" ".join(lemmas_noticia))
        print(f"  - Extraídos {len(lemmas_noticia)} lemas de {conllu_file}")
    else:
        print(f"  - No se pudieron extraer lemas de {conllu_file}")
       # handlear la ausencia de lenmas
        corpus_para_tfidf.append("") # Añadir string vacío si falla

# Verificar que tenemos datos para procesar
if not any(corpus_para_tfidf):
    print("\nError: No se pudieron obtener lemas de ningún archivo. No se puede continuar con TF-IDF.")
elif len([c for c in corpus_para_tfidf if c]) < 2:
     print("\nAdvertencia: Se necesita al menos 2 documentos con contenido para calcular IDF. Los resultados pueden no ser significativos.")
     # Podrías optar por no ejecutar TF-IDF en este caso
else:
    print("\nLemas listos para análisis TF-IDF.")

```


```python
#nltk.download('stopwords', quiet=True) # Bajamos las stopwords de NLTK
```


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np

tfidf_matrix = None
feature_names = []
vectorizer = None

# Solo proceder si tenemos al menos un documento con contenido
# (Idealmente, necesitamos al menos 2 para que el idf tenga sentido)
if any(corpus_para_tfidf) and len([c for c in corpus_para_tfidf if c]) >= 1:
    print("Calculando TF-IDF...")

    stop_words_es = stop_words_es = stopwords.words('spanish')
    print(f"  Se usarán {len(stop_words_es)} stop words de NLTK.")
        # También podemos agregar sw:
        # stop_words_es.extend(['palabra_extra1', 'palabra_extra2'])
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm='l2', stop_words=stop_words_es)
    tfidf_matrix = vectorizer.fit_transform(corpus_para_tfidf)
    feature_names = vectorizer.get_feature_names_out()

    print(f"Matriz TF-IDF calculada: {tfidf_matrix.shape[0]} documentos, {tfidf_matrix.shape[1]} lemas únicos.")
else:
    print("No hay suficientes datos válidos para calcular TF-IDF.")
```


```python
def get_top_tfidf_lemmas(doc_index, tfidf_matrix, feature_names, top_n=10):
    """Obtiene los top N lemas para un documento específico."""
    # Asegurarse de que la matriz y los nombres existan
    if tfidf_matrix is None or not feature_names.any():
         return []

    doc_scores = tfidf_matrix[doc_index].toarray().flatten()
    # Obtenemos índices ordenados de mayor a menor score
    sorted_indices = np.argsort(doc_scores)[::-1]
    # Seleccionar los top N índices (asegurarse de no exceder el número de features)
    top_indices = sorted_indices[:min(top_n, len(feature_names))]
    # Obtengo tmb  los lemas y scores correspondientes
    top_lemmas_scores = [(feature_names[i], doc_scores[i]) for i in top_indices if doc_scores[i] > 0] # Solo incluir si score > 0
    return top_lemmas_scores

# Mostrar los resultados si se calculó TF-IDF.
if tfidf_matrix is not None:
    print("\n--- Lemas más Característicos por Noticia (según TF-IDF) ---")
    for i, filepath in enumerate(nombres_archivos_conllu):
        # Solo mostrar si el documento original tenía contenido
        if corpus_para_tfidf[i]:
            print(f"\nNoticia: {filepath}")
            top_lemmas = get_top_tfidf_lemmas(i, tfidf_matrix, feature_names, top_n=15) # Pedir 15, por ejemplo
            if top_lemmas:
                for lemma, score in top_lemmas:
                    print(f"  - {lemma:<20} (Score: {score:.4f})")
            else:
                print("  No se encontraron lemas con score TF-IDF > 0.")
        else:
             print(f"\nNoticia: {filepath} (Archivo vacío o no procesado, se omite)")
```


```python

```

{% include additional_content.html %}

