import matplotlib.pyplot as plt
import json
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

def plot_linear_function(m=2, b=1, x_range=(-10, 10), num_points=100, show_data=False, noise_std=1.0):
    """
    Grafica una función lineal y opcionalmente datos simulados con ruido.

    Parámetros:
    - m (float): pendiente de la función lineal.
    - b (float): intercepto de la función lineal.
    - x_range (tuple): rango de valores x (mín, máx).
    - num_points (int): número de puntos para la gráfica.
    - show_data (bool): si es True, se generan puntos de datos con ruido para simular datos de regresión.
    - noise_std (float): desviación estándar del ruido gaussiano (solo si show_data es True).
    """
    # Generamos valores de x en el rango indicado
    x = np.linspace(x_range[0], x_range[1], num_points)
    # Calculamos la función lineal y = m*x + b
    y = m * x + b

    plt.figure(figsize=(8, 6))
    # Graficamos la función lineal
    plt.plot(x, y, label=f'Función lineal: y = {m}x + {b}', color='blue')
    
    if show_data:
        # Agregamos ruido gaussiano a los valores de y para simular datos
        y_noise = y + np.random.normal(0, noise_std, size=num_points)
        plt.scatter(x, y_noise, color='red', alpha=0.6, label='Datos con ruido')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gráfica de una función lineal')
    plt.legend()
    plt.grid(True)
    plt.show()


def load_json_data(path):
    """
    Carga datos desde un archivo JSON y los devuelve como un objeto Python.
    """
    with open(path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data


def save_json(data, file):
    """
    Guarda un objeto Python en un archivo JSON.
    """
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
        
def train_sentiment_classifier():
    """
    Entrena un modelo de clasificación binaria para análisis de sentimientos utilizando el dataset IMDb.
    
    Devuelve:
        - pipeline: el modelo entrenado (pipeline de scikit-learn).
        - report: un string con el reporte de clasificación.
    """
    
    # Cargamos el dataset "imdb" y seleccionamos una muestra para entrenamiento y prueba
    dataset = load_dataset("imdb")
    train_data = dataset['train'].shuffle(seed=42).select(range(5000))  # Muestra de 5000 ejemplos
    test_data = dataset['test'].shuffle(seed=42).select(range(1000))      # Muestra de 1000 ejemplos

    # Extraemos los textos y las etiquetas
    X_train = train_data['text']
    y_train = train_data['label']
    X_test = test_data['text']
    y_test = test_data['label']

    # Creamos un pipeline que vectoriza el texto con TF-IDF y entrena un modelo de regresión logística
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=10000),
        LogisticRegression(solver='liblinear')
    )

    # Entrenamos el modelo
    pipeline.fit(X_train, y_train)

    # Realizamos predicciones y generamos un reporte de clasificación
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred)

    return pipeline, report
