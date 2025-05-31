### Sugerencias de uso de la Notebook: 
- Sugerimos 'Abrir en Colab' y realizar una copia del cuaderno antes de usarlo.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1G0gjd9M-pKM-MmKCCbYGG6fByHCPbotZ?usp=sharing)

# Parsers basados en constituyentes


```python
import nltk
!pip install --user bllipparser
```


```python
import bllipparser
from bllipparser import RerankingParser                             #Importa el parser
from bllipparser.ModelFetcher import download_and_install_model     # Descarga e instala el "modelo"

model_dir = download_and_install_model('WSJ', 'tmp/models')         #Crea una variable con el "modelo"
rrp = RerankingParser.from_unified_model_dir(model_dir)
```


```python
oracion2 = "El sur de Alemania lucha contra el poder de la lluvia."
parseo = rrp.simple_parse(oracion2)
```


```python
print(parseo)
type(parseo)
```

# Parsers de dependencias

## Spacy


```python
!pip install spacy
!python -m spacy download es_core_news_sm
!python -m spacy download es_core_news_md
```


```python
import spacy
from nltk import Tree
from spacy import displacy

def gramaticadependencias(sentence, model):       #Define la función
    nlp = spacy.load(model)    #Carga el modelo entrenado
    doc = nlp(sentence)                    #define una variable doc con la oración procesada por el modelo
    for token in doc:
        print(token.text, token.pos_, token.dep_, #token.head.text, token.head.pos_,
            [child for child in token.children])
    displacy.render(doc, style='dep', jupyter=True)

modelsm = 'es_core_news_sm'
modelmd = 'es_core_news_md'
```


```python
sentence = 'la mañana está fría para andar paseando'
gramaticadependencias(sentence, modelmd)
```

## Stanza


```python
!pip install stanza
import stanza
stanza.download('es') # Baja el modelo para el español
```


```python
nlp = stanza.Pipeline('es') # Inicializa el modelo de español (con su pipeline de anotación)
nlp
```


```python
sentence = 'la mañana está muy fría para andar paseando'
doc = nlp(sentence) # Anota una oración
doc
```


```python
doc.sentences[0].print_dependencies()
```


```python
doc.sentences[0].dependencies
```

[Página de Universal Dependencies](https://universaldependencies.org/format.html)
