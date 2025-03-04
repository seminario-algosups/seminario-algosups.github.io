import matplotlib.pyplot as plt
import json
import numpy as np

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
    with open(path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data


def save_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)