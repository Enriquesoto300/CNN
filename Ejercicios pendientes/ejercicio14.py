def es_primo(n):
    """
    Verifica si un número es primo.
    
    Parámetros:
    n (int): Número entero a verificar.

    Retorna:
    bool: True si es primo, False si no.
    """
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def factorial(n):
    """
    Calcula el factorial de un número de forma recursiva.

    Parámetros:
    n (int): Número entero no negativo.

    Retorna:
    int: Factorial de n.
    """
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

def celsius_a_fahrenheit(c):
    """
    Convierte grados Celsius a Fahrenheit.

    Parámetros:
    c (float): Temperatura en grados Celsius.

    Retorna:
    float: Temperatura en Fahrenheit.
    """
    return (c * 9/5) + 32

# Verificación para pruebas independientes
if __name__ == "__main__":
    print("Probando funciones dentro de mis_utilidades.py...\n")
    print("¿7 es primo?", es_primo(7))
    print("Factorial de 5:", factorial(5))
    print("30°C en Fahrenheit:", celsius_a_fahrenheit(30))

