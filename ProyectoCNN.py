import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Precisión en datos de prueba: {test_acc:.4f}')


def predecir_imagen(ruta_imagen):
    
    img = Image.open(ruta_imagen).convert("L")
    
    img = img.resize((28, 28))
    
    img_array = np.array(img)
    
    img_array = 255 - img_array  
    
    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    
    plt.imshow(img_array[0].reshape(28, 28), cmap="gray")
    plt.title("Imagen cargada")
    plt.show()
    
    prediccion = model.predict(img_array)
    print(f"Predicción de la red: {np.argmax(prediccion)}")

predecir_imagen(r"C:\Users\Trans\Downloads\CNN\img\SieteDeMNIST.png")

