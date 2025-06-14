### Sugerencias de uso de la Notebook: 
- Sugerimos 'Abrir en Colab' y realizar una copia del cuaderno antes de usarlo.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seminario-algosups/seminario-algosups.github.io/blob/master/Clase-13/entities-2.ipynb)

# Entrenando un Reconocedor de Entidades Nombradas (NER) con spaCy



**Objetivo:** Entrenar un modelo de `spaCy` para que reconozca tres tipos de entidades específicas en noticias de diarios digitales:
*   **PERSON**: Nombres de personas.
*   **GPE** (Geopolitical Entity): Lugares geográficos como ciudades, estados o países.
*   **DATE**: Fechas completas o parciales.

**El Proceso:**
1.  **Recolección de Datos:** Usaremos textos de noticias que ustedes mismos traerán.
2.  **Anotación:** "Etiquetaremos" manualmente las entidades en nuestros textos para enseñarle al modelo qué debe buscar.
3.  **Preparación:** Convertiremos nuestras anotaciones al formato que `spaCy` necesita para aprender.
4.  **Entrenamiento:** Ejecutaremos el proceso de entrenamiento de `spaCy`.
5.  **Evaluación:** Probaremos nuestro nuevo modelo "a medida" con un texto nuevo.




```python
# =============================================================================
# PASO 1: INSTALACIÓN Y CONFIGURACIÓN
# =============================================================================
# Primero, instalamos spaCy y descargamos un modelo base en español.
# Usaremos un modelo pequeño ("sm") como punto de partida. Nuestro modelo
# aprenderá de él y se especializará con nuestros datos.

#!pip install -U spacy
!python -m spacy download es_core_news_sm

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch, compounding
import random

print("✅ Librerías y modelo base instalados y cargados.")
```


```python
# =============================================================================
# FUNCIÓN DE PARSING: Convierte el formato [texto](ETIQUETA) a formato spaCy
# =============================================================================
import re

def parse_annotated_text(annotated_text):
    """
    Parsea un texto con anotaciones estilo Markdown [texto](ETIQUETA) y lo 
    convierte al formato de datos de entrenamiento de spaCy.

    Args:
        annotated_text (str): El texto con anotaciones.
        Ej: "Viajé a [Madrid](GPE) con [Juan Pérez](PERSON)."

    Returns:
        tuple: Una tupla con (texto_limpio, {"entities": [...]}), o None si hay error.
    """
    
    clean_text = ""
    entities = []
    last_index = 0
    
    # Regex para encontrar patrones como [texto de la entidad](ETIQUETA)
    # Captura dos grupos: 1) el texto, 2) la etiqueta
    pattern = re.compile(r"\[(.+?)\]\((.+?)\)")

    # Iteramos sobre todas las coincidencias encontradas en el texto
    for match in pattern.finditer(annotated_text):
        entity_text = match.group(1)
        entity_label = match.group(2)
        
        # Añadimos el texto que hay entre la última entidad y la actual
        clean_text += annotated_text[last_index:match.start()]
        
        # Calculamos los índices de inicio y fin de la entidad en el texto limpio
        start_index = len(clean_text)
        clean_text += entity_text
        end_index = len(clean_text)
        
        entities.append((start_index, end_index, entity_label))
        
        # Actualizamos el índice de la última posición procesada
        last_index = match.end()

    # Añadimos el resto del texto que pueda quedar después de la última entidad
    clean_text += annotated_text[last_index:]
    #print(clean_text, {"entities": entities})
    return (clean_text, {"entities": entities})

def parse_annotated_text_v2(annotated_text, debug=False):
    clean_text, entities, last_index = "", [], 0
    pattern = re.compile(r"\[(.+?)\]\((.+?)\)")
    if debug: print(f"--- Depurando texto: \"{annotated_text[:50]}...\" ---")
    for i, match in enumerate(pattern.finditer(annotated_text)):
        entity_text, entity_label = match.group(1), match.group(2).upper()
        prefix = annotated_text[last_index:match.start()]
        clean_text += prefix
        start_index = len(clean_text)
        clean_text += entity_text
        end_index = len(clean_text)
        entities.append((start_index, end_index, entity_label))
        last_index = match.end()
        if debug:
            print(f"\nMatch #{i+1}: '{entity_text}' ({entity_label})")
            print(f"  - Prefijo: '{prefix}'")
            print(f"  - Índices: ({start_index}, {end_index})")
            print(f"  - Texto limpio parcial: '{clean_text}'")
    suffix = annotated_text[last_index:]
    clean_text += suffix
    if debug: 
        print(f"\nSufijo: '{suffix}'")
        print(f"--- Fin ---")
    return (clean_text, {"entities": entities})
```


```python

```


```python

```

## PASO 2: Los Datos 

Aquí es donde comienza la magia. En la siguiente celda de código, hay una lista de Python llamada `TEXTOS_NOTICIAS`.

**Instrucciones:**
1.  Buscar 2 o 3 artículos cortos de noticias en diarios digitales.
2.  Copiar el texto de cada artículo.
3.  Pegar cada texto como un string dentro de la lista `TEXTOS_NOTICIAS`. Asegurarse de que cada texto esté entre comillas `""" """` y separado por una coma.

He dejado un ejemplo para que veamos el formato...




```python
# =============================================================================
# PASO 2: LOS DATOS
# =============================================================================
# ACCIÓN REQUERIDA: 
# Pegar los textos de tus noticias.
# Reemplazar el ejemplo con tus propios textos.

TEXTOS_NOTICIAS = [
    """El futbolista Lionel Messi llegó a París el 10 de agosto de 2021 para firmar su contrato con el PSG. La presentación oficial se realizó en el estadio Parque de los Príncipes.""",
    
    """La cumbre del G20 se celebrará en Río de Janeiro durante el mes de noviembre. Se espera la asistencia de líderes como Joe Biden y Pedro Sánchez.""",

    """La NASA anunció el pasado 3 de marzo que la misión Artemis II volverá a la Luna en 2025. El astronauta Reid Wiseman será el comandante."""
]

# Mostramos los textos cargados para verificar
print(f"Se cargaron {len(TEXTOS_NOTICIAS)} textos para anotar.")
for i, texto in enumerate(TEXTOS_NOTICIAS):
    print(f"  Texto {i+1}: {texto[:80]}...") # Mostramos los primeros 80 caracteres
```


```python

```


## PASO 3: La Anotación

Este es el paso más importante. Vamos a "etiquetar" las entidades en nuestros textos usando un formato mucho más sencillo e intuitivo

**Instrucciones**
1.  Tomar los textos de tus noticias que pegaste en el `PASO 2`.
2.  En la celda de abajo, en la lista `ANNOTATED_TEXTS`, copiar y pegar cada texto.
3.  Directamente sobre el texto, vamos a "envolvre" cada entidad con el formato `[texto de la entidad](ETIQUETA)`.
    *   Usa `PERSON` para personas.
    *   Usa `GPE` para lugares (ciudades, países).
    *   Usa `DATE` para fechas.

**Ejemplo:**
*   **Texto Original:** `"El futbolista Lionel Messi llegó a París el 10 de agosto de 2021."`
*   **Texto Anotado:** `"El futbolista [Lionel Messi](PERSON) llegó a [París](GPE) el [10 de agosto de 2021](DATE)."`




```python
# =============================================================================
# PASO 3: ANOTACIÓN DE DATOS 
# =============================================================================
# ACCIÓN REQUERIDA: Anotar tus textos usando el formato [entidad](ETIQUETA)

ANNOTATED_TEXTS = [
    "El futbolista [Lionel Messi](PERSON) llegó a [París](GPE) el [10 de agosto de 2021](DATE) para firmar su contrato con el PSG. La presentación oficial se realizó en el estadio Parque de los Príncipes.",
    
    "La cumbre del G20 se celebrará en [Río de Janeiro](GPE) durante el mes de [noviembre](DATE). Se espera la asistencia de líderes como [Joe Biden](PERSON) y [Pedro Sánchez](PERSON).",

    "La NASA anunció el pasado [3 de marzo](DATE) que la misión Artemis II volverá a la [Luna](GPE) en [2025](DATE). El astronauta [Reid Wiseman](PERSON) será el comandante."
]


# =============================================================================
# CONVERSIÓN AUTOMÁTICA
# =============================================================================
# Ahora usamos nuestra nueva función para crear los datos de entrenamiento

TRAIN_DATA = []
print("Procesando anotaciones...")
for text in ANNOTATED_TEXTS:
    parsed_data = parse_annotated_text(text)
    TRAIN_DATA.append(parsed_data)

print("✅ Anotaciones procesadas y convertidas al formato de spaCy.")

# Verifiquemos el resultado de la conversión para el primer texto
print("\nEjemplo de conversión:")
print("Texto anotado original:")
print(f"  > {ANNOTATED_TEXTS[0]}")
print("\nConvertido a formato spaCy:")
print(f"  > {TRAIN_DATA[0]}")
```


```python
MODO_DEBUG = True 

for text in ANNOTATED_TEXTS:
    parsed_data = parse_annotated_text_v2(text, debug=MODO_DEBUG)
    TRAIN_DATA.append(parsed_data)
    if MODO_DEBUG: print("-" * 50)

print("\n✅ Anotaciones procesadas y convertidas al formato de spaCy.")

# Verifiquemos el resultado final de la conversión para el primer texto
print("\nEjemplo del resultado final (primer texto):")
print(TRAIN_DATA[0])
```

## PASO 4: Preparación y Entrenamiento del Modelo

¡Ya casi estamos! Ahora vamos a tomar nuestros datos de entrenamiento y usarlos para actualizar el modelo base de `spaCy`.

**¿Qué sucede aquí?**
1.  **Cargar modelo base:** Partimos de `es_core_news_sm`, que ya sabe mucho sobre español.
2.  **Añadir etiquetas:** Le decimos al modelo que ahora debe reconocer `PERSON`, `GPE` y `DATE`.
3.  **Bucle de entrenamiento:** Le mostramos nuestros ejemplos anotados una y otra vez (en "épocas" o `epochs`), y el modelo ajusta sus "pesos" internos para minimizar el error. Es como estudiar para un examen: cada repaso refuerza el conocimiento.
4.  **Guardar el modelo:** Al final, guardamos la nueva versión mejorada del modelo en una carpeta.

Este proceso puede tardar unos minutos.


```python
# =============================================================================
# PASO 4: ENTRENAMIENTO
# =============================================================================

# Cargamos el modelo pre-entrenado
nlp = spacy.load("es_core_news_sm")

# Obtenemos el componente de 'ner' de la pipeline
ner = nlp.get_pipe("ner")

# Añadimos las nuevas etiquetas al componente NER
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Desactivamos otros componentes de la pipeline que no vamos a entrenar
# para que el entrenamiento sea más eficiente
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# --- Bucle de entrenamiento ---
epochs = 20 # Número de veces que le mostraremos los datos al modelo
print("Iniciando entrenamiento...")

with nlp.disable_pipes(*unaffected_pipes):
    for epoch in range(epochs):
        random.shuffle(TRAIN_DATA)
        losses = {}
        
        # Dividimos los datos en lotes (batches) para un mejor rendimiento
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        
        for batch in batches:
            # Creamos objetos Example para el lote actual
            examples = []
            for text, annot in batch:
                print(annot)
                examples.append(Example.from_dict(nlp.make_doc(text), annot))
            
            # Actualizamos el modelo con los ejemplos
            nlp.update(
                examples,
                drop=0.4,  # Dropout - para evitar sobreajuste
                losses=losses,
            )
        
        print(f"Época {epoch+1}/{epochs} - Pérdida (Loss): {losses['ner']:.4f}")

print("✅ ¡Entrenamiento completado!")

# Guardamos el modelo entrenado en una nueva carpeta
output_dir = "./ner_model_custom"
nlp.to_disk(output_dir)
print(f"Modelo guardado en: {output_dir}")
```

## PASO 5: ¡A Probar Nuestro Modelo!

Llegó el momento de la verdad. Vamos a cargar nuestro modelo recién entrenado y a probarlo con una frase que no haya visto antes.

Veremos si es capaz de identificar correctamente las personas, lugares y fechas. Usaremos `displaCy`, una herramienta de visualización de `spaCy` que colorea las entidades de forma muy clara.


```python
# =============================================================================
# PASO 5: EVALUACIÓN Y PRUEBA
# =============================================================================
from spacy import displacy

# Cargamos nuestro modelo personalizado desde la carpeta donde lo guardamos
print("Cargando el modelo personalizado...")
nlp_custom = spacy.load(output_dir)

# ✍️ ACCIÓN REQUERIDA: Escribir aquí un texto nuevo para probar el modelo.
# Intentar que sea similar a los que usaste para entrenar.
texto_de_prueba = "Lio Messi viajará a Madrid durante Noviembre para reunirse con Pedro Sanchez." 
#texto_de_prueba = "El ex presidente Barack Obama visitará Barcelona en 2024 para dar una conferencia sobre cambio climático."


print(f"\nProbando el modelo con el texto:\n'{texto_de_prueba}'\n")

# Procesamos el texto con nuestro modelo
doc = nlp_custom(texto_de_prueba)

# Imprimimos las entidades encontradas
print("Entidades encontradas:")
if not doc.ents:
    print("-> No se encontraron entidades.")
else:
    for ent in doc.ents:
        print(f"  - Texto: '{ent.text}', Etiqueta: '{ent.label_}'")


```


```python
from IPython.display import display, HTML

print("\nVisualización:")

# 1. Generamos el código HTML para la visualización (sin intentar mostrarlo automáticamente)
html = displacy.render(doc, style="ent", jupyter=False)

# 2. Mostramos el HTML generado usando la función display() de IPython
display(HTML(html))
```


```python

```


```python

```

## Conclusiones y Próximos Pasos

¡Felicitaciones! Han recolectado, anotado y entrenado con éxito su propio modelo de reconocimiento de entidades.

**¿Qué hemos logrado?**
*   Hemos visto que los modelos de IA no son "cajas negras mágicas". Se pueden adaptar y mejorar con datos específicos.
*   Entendimos la importancia crucial de la **calidad de los datos y las anotaciones**. Si las anotaciones son incorrectas, el modelo aprenderá mal.
*   Hemos creado una herramienta que ahora está especializada en encontrar personas, lugares y fechas en el tipo de texto que le proporcionamos.

**Posibles Próximos Pasos:**
*   **Añadir más datos:** Con solo 3 textos, el modelo es frágil. Con 30, 50 o 100 textos, se volvería mucho más robusto.
*   **Añadir más etiquetas:** ¿Qué tal anotar `ORG` para organizaciones (como "PSG", "NASA")?
*   **Evaluar formalmente:** En proyectos reales, separamos los datos en un conjunto de entrenamiento y otro de evaluación para medir numéricamente qué tan bueno es nuestro modelo (precisión, recall, F1-score).


