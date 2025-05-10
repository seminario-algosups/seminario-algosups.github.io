### Sugerencias de uso de la Notebook: 
- Sugerimos 'Abrir en Colab' y realizar una copia del cuaderno antes de usarlo.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jHG8yChrcPLbymByLHswzJdeTf_3u2Bs?usp=sharing)

```python
import nltk # Importa la librería NLTK
from nltk.tag import pos_tag # Importa los paquetes de tag
from nltk.tokenize import word_tokenize # Importa el tokenizador de nltk
nltk.download('brown')
nltk.download('punkt') # BA
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('tagsets')
nltk.download('treebank')
from nltk.corpus import brown
```


```python
# ¿Qué pasa con el español?
sentence = "Yo lo vi ayer en la panadería"
pos_tag(word_tokenize(sentence),lang='es')
```


```python
# ¿Existen realmente las clases de palabras?
text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
text.similar('run')
```


```python
sentence = "I found a crying baby in the surrounding area."
#print(word_tokenize(sentence))
pos_tag(word_tokenize(sentence))
```


```python
# http://www.natcorp.ox.ac.uk/docs/gramtag.html
nltk.help.upenn_tagset()
```


```python
 nltk.corpus.brown.tagged_words()[20:40]
```

Brown: Manual de anotación del corpus de Brown

BNC: Manual de anotación del BNC

BNC Otro manual de anotación del BNC

- Ver el texto de modelo en la página.
- Crear un txt localmente (se puede hacer también en modalidad manuscrita en un cuaderno).
- Copiar dos párrafos de una noticia cualquiera.
- Separar los signos de puntuación.
- Indicar los datos que se juzguen relevantes acerca de la fuente del texto anteponiendo a la línea dos numerales. Por ejemplo:
```
## www.unawebcualquiera.com/untextocualquieradelquesaqueeltexto
```

- Usar las categorías del tagset que se puede encontrar [aquí](http://www.natcorp.ox.ac.uk/docs/gramtag.html)
- Agregar las categorías inmediatamente después de cada palabra usando barra inclinada para separar la palabra de la categoría.
- En el caso de signos de puntuación pegados, separarlos.
- Este tagset no está pensado para el español y al aplicarlo a esta lengua surgen inadecuaciones. Anotar todas las inadecuaciones que surjan y pensar posibles modificaciones al tagset para solucionarlas.



```python
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
tag_fd.most_common()
```


```python
word_tag_pairs = nltk.bigrams(brown_news_tagged)
noun_preceders = [a[1] for (a, b) in word_tag_pairs if b[1] == 'ADJ']
fdist = nltk.FreqDist(noun_preceders)
[tag for (tag, _) in fdist.most_common()]
```


```python
wsj = nltk.corpus.treebank.tagged_words(tagset='universal')
word_tag_fd = nltk.FreqDist(wsj)
[wt[0] for (wt, _) in word_tag_fd.most_common() if wt[1] == 'NOUN']
```


```python
# Ver documentación en https://spacy.io/models/es
!python -m spacy download es_core_news_sm
!python -m spacy download es_core_news_md
```


```python
import spacy

nlp = spacy.load("es_core_news_sm")
sent_es = "Yo soy aburrido"
doc = nlp(sent_es)

for token in doc:
    print(token.text, token.lemma_, token.pos_)

```


```python
print(*[f'word: {token.text}\tlemma: {token.lemma_}\tpos: {token.pos_}' for token in doc], sep='\n')
```


```python
!pip install stanza
```


```python
import stanza
nlp = stanza.Pipeline('en', processors='tokenize,pos')
```


```python
en_test_= "Jorge Luis Borges was born in Argentina. He died in Geneva in 1986. He has writen a lots of book, but no novel."
doc = nlp(en_test_)
sent_list = [sent.text for sent in doc.sentences]
print([f'{word.text}/{word.pos}/{word.upos}' for sent in doc.sentences for word in sent.words])
print(*[f'word: {word.text}\tpos: {word.pos}\tupos: {word.upos}' for sent in doc.sentences for word in sent.words], sep='\n')
```


```python
nlp = stanza.Pipeline('es', processors='tokenize,pos')
```


```python
es_test_= "Jorge Luis Borges nació en Argentina. Murió en Ginebra en 1986. Escribió muchos libros, pero ninguna novela."
doc = nlp(es_test_)
sent_list = [sent.text for sent in doc.sentences]
print([f'{word.text}/{word.pos}/{word.upos}' for sent in doc.sentences for word in sent.words])
print(*[f'word: {word.text}\tpos: {word.pos}\tupos: {word.upos}' for sent in doc.sentences for word in sent.words], sep='\n')
```

Anotar dos oraciones de una noticia según la distribución que se haga en clase.

Las categorías deben sacarse de [acá](https://universaldependencies.org/es/pos/index.html)
