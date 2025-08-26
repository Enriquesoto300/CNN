import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ==========================
# 1. Cargar y preparar MNIST
# ==========================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizar
x_train = x_train / 255.0
x_test = x_test / 255.0

# Expandir dimensiones para el canal
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# ==========================
# 2. Definir modelo CNN
# ==========================
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 dígitos
])

# Compilar
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ==========================
# 3. Entrenar modelo
# ==========================
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluar
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Precisión en datos de prueba: {test_acc:.4f}')

# ==========================
# 4. Probar con imagen propia
# ==========================
def predecir_imagen(ruta_imagen):
    # Abrir imagen y convertir a escala de grises
    img = Image.open(ruta_imagen).convert("L")
    
    # Redimensionar a 28x28
    img = img.resize((28, 28))
    
    # Convertir a array
    img_array = np.array(img)
    
    # Invertir colores (MNIST es fondo negro con dígito blanco)
    img_array = 255 - img_array  
    
    # Normalizar
    img_array = img_array / 255.0
    
    # Dar forma (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Mostrar imagen
    plt.imshow(img_array[0].reshape(28, 28), cmap="gray")
    plt.title("Imagen cargada")
    plt.show()
    
    # Predicción
    prediccion = model.predict(img_array)
    print(f"Predicción de la red: {np.argmax(prediccion)}")

# ==========================
# 5. Usar tu propia imagen
# ==========================
# Cambia "mi_digito.png" por la ruta de tu archivo
predecir_imagen(r"C:\Users\Trans\Downloads\Programacion\red neuronal 2\SieteDeMNIST.png")

