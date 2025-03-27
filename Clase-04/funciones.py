import nltk
import string
from nltk.corpus import stopwords
from unidecode import unidecode
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.probability import FreqDist
from wordcloud import WordCloud

# Download stopwords if not already present
stopword_es = nltk.corpus.stopwords.words('spanish')

def preprocess_text(text):
    """
    Preprocess Spanish text:
    - Converts to lowercase
    - Removes accents
    - Removes punctuation
    - Removes stopwords
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove accents
    text = unidecode(text)
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove stopwords
    stop_words = set(stopwords.words('spanish'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    
    return " ".join(filtered_words)


def plot_text_word_matrix(texts):
        # Vectorizar los textos
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()

    # Convertir la matriz dispersa a densa
    X_dense = X.toarray()

    # Ajustar el tamaño de la figura para mejor visibilidad
    fig, ax = plt.subplots(figsize=(max(12, len(words) // 2), 8))  # Ancho dinámico

    # Graficar la matriz
    cax = ax.matshow(X_dense, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Configurar etiquetas del eje Y
    ax.set_yticks(np.arange(len(texts)))
    ax.set_yticklabels([f'Text {i+1}' for i in range(len(texts))])

    # Configurar etiquetas del eje X
    ax.set_xticks(np.arange(len(words)))

    # Mostrar solo algunas etiquetas del eje X si hay demasiadas palabras
    if len(words) > 50:  # Umbral de cantidad de palabras
        step = len(words) // 50
        words = [word if i % step == 0 else "" for i, word in enumerate(words)]

    ax.set_xticklabels(words, rotation=90, ha='right')

    plt.xlabel('Vocabulario')
    plt.ylabel('Textos')
    plt.title('Matriz de Frecuencia de Palabras')
    plt.show()
    
    
def plot_bag_of_words(sentences):
    all_words = []

    for sentence in sentences:
        norm_sentence = preprocess_text(sentence)
        all_words.extend([word for word in norm_sentence.split()])

    wordcloud = WordCloud().generate(" ".join(all_words))

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
    
def plot_tfif_matrix(X, texts, words):
    # Convertir la matriz dispersa a densa
    X_dense = X.toarray()

    # Ajustar tamaño de la figura según la cantidad de términos
    fig, ax = plt.subplots(figsize=(max(12, len(words) // 2), 8))

    # Graficar la matriz con TF-IDF como intensidad del color
    cax = ax.matshow(X_dense, cmap='Blues', aspect='auto')
    fig.colorbar(cax)

    # Agregar los valores numéricos dentro de las celdas
    for i in range(X_dense.shape[0]):  # Recorre los textos (filas)
        for j in range(X_dense.shape[1]):  # Recorre los términos (columnas)
            value = X_dense[i, j]
            if value > 0:  # Solo mostrar valores distintos de 0
                ax.text(j, i, f"{value:.2f}", ha='center', va='center', fontsize=8, color='black')

    # Configurar etiquetas del eje Y
    ax.set_yticks(np.arange(len(texts)))
    ax.set_yticklabels([f'Text {i+1}' for i in range(len(texts))])

    # Configurar etiquetas del eje X
    ax.set_xticks(np.arange(len(words)))

    # Reducir etiquetas en el eje X si hay demasiadas palabras
    if len(words) > 50:
        step = len(words) // 50
        words = [word if i % step == 0 else "" for i, word in enumerate(words)]

    ax.set_xticklabels(words, rotation=90, ha='right')

    plt.xlabel('Términos')
    plt.ylabel('Textos')
    plt.title('Matriz de Frecuencia TF-IDF')
    plt.show()
    
    