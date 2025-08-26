# Programa: Uso de paquetes con pip
# Librería utilizada: pyjokes
# Descripción:
# pyjokes es una librería de Python que permite obtener chistes (generalmente relacionados con programadores o geeks)
# de manera aleatoria. Se pueden imprimir en consola y son útiles para practicar el uso de pip y librerías externas.

import pyjokes  # Importamos la librería instalada con pip

def mostrar_chiste():
    """Obtiene un chiste aleatorio usando pyjokes"""
    chiste = pyjokes.get_joke()
    print(f"Chiste aleatorio: {chiste}")

def main():
    print("=== Programa de prueba con pyjokes ===")
    mostrar_chiste()
    print("Fin del programa.")

if __name__ == "__main__":
    main()
