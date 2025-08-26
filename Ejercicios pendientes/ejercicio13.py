import numpy as np

# Función para capturar una matriz desde el usuario
def leer_matriz(nombre):
    try:
        filas = int(input(f"¿Cuántas filas tiene la matriz {nombre}? "))
        columnas = int(input(f"¿Cuántas columnas tiene la matriz {nombre}? "))
        print(f"Introduce los elementos de la matriz {nombre} fila por fila, separados por espacios:")

        matriz = []
        for i in range(filas):
            fila = input(f"Fila {i + 1}: ").strip().split()
            if len(fila) != columnas:
                print("Número incorrecto de elementos. Intenta de nuevo.")
                return leer_matriz(nombre)
            matriz.append([float(x) for x in fila])
        return np.array(matriz)
    except ValueError:
        print("Entrada inválida. Intenta de nuevo.")
        return leer_matriz(nombre)

# Función para mostrar el menú y obtener la opción del usuario
def mostrar_menu():
    print("\nCalculadora Matricial - Selecciona una operación:")
    print("1. Suma de matrices")
    print("2. Resta de matrices")
    print("3. Multiplicación de matrices")
    print("4. Transposición de una matriz")
    print("5. Salir")
    opcion = input("Elige una opción (1-5): ")
    return opcion

# Función principal que controla el flujo del programa
def calculadora_matricial():
    while True:
        opcion = mostrar_menu()

        if opcion == "1":  # Suma
            print("\nSUMA DE MATRICES")
            A = leer_matriz("A")
            B = leer_matriz("B")
            if A.shape == B.shape:
                print("\nResultado:\n", A + B)
            else:
                print("Error: Las matrices deben tener las mismas dimensiones para sumarse.")

        elif opcion == "2":  # Resta
            print("\nRESTA DE MATRICES")
            A = leer_matriz("A")
            B = leer_matriz("B")
            if A.shape == B.shape:
                print("\nResultado:\n", A - B)
            else:
                print("Error: Las matrices deben tener las mismas dimensiones para restarse.")

        elif opcion == "3":  # Multiplicación
            print("\nMULTIPLICACIÓN DE MATRICES")
            A = leer_matriz("A")
            B = leer_matriz("B")
            if A.shape[1] == B.shape[0]:
                print("\nResultado:\n", np.dot(A, B))
            else:
                print("Error: El número de columnas de A debe coincidir con el número de filas de B.")

        elif opcion == "4":  # Transposición
            print("\nTRANSPOSICIÓN DE UNA MATRIZ")
            A = leer_matriz("A")
            print("\nMatriz original:\n", A)
            print("\nMatriz transpuesta:\n", A.T)

        elif opcion == "5":  # Salir
            print("Gracias por usar la calculadora matricial. ¡Hasta luego!")
            break

        else:
            print("Opción no válida. Intenta de nuevo.")

# Ejecutar el programa
calculadora_matricial()
