

def pasar_a_mayusculas(texto: str) -> str:
    """
    Convierte un texto a mayúsculas.

    Args: 
        texto (str): Texto a convertir.

    Returns:
        str: El texto convertido a mayúsculas.

    Raises:
        TypeError: Si el argumento no es una cadena de texto.
    """
    if not isinstance(texto, str):
        raise TypeError("El argumento debe ser una cadena de texto (str).")
    
    return texto.upper()


def sumar_elementos_en_lista(lista: list) -> int:
    
    return sum(lista)