# Reconocimiento de Dígitos con Red Neuronal Convolucional

## Este proyecto implementa una red neuronal convolucional (CNN) usando **TensorFlow** y **Keras** para reconocer dígitos escritos a mano utilizando el dataset **MNIST**.  
### Además, permite probar imágenes personalizadas para evaluar el rendimiento del modelo.

## Requisitos
### Antes de ejecutar el proyecto, asegúrate de tener instalado:


* Python 3.8 o superior  
- TensorFlow  
- Matplotlib  
- Numpy  
- Pillow    

**Puedes instalar las dependencias con:**  
```bash
pip install tensorflow matplotlib numpy pillow
```
## __Ejecución del proyecto__

**1. Clona el repositorio:** 


```git clone https://github.com/Enriquesoto300/No-vuelvo-a-faltar..git```

**2. Ingresa al directorio:**

```cd No-vuelvo-a-faltar```

**3. Ejecuta el programa:**

```python CNN.py```


## Explicación del código

### 1. Importar librerías

![This is an alt text.](C:\Users\Trans\Downloads\CNN\README\Librerias.png "This is a sample image.")
```import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
```


* TensorFlow/Keras: Construcción y entrenamiento de la red neuronal.
* Matplotlib: Visualización de imágenes.
* NumPy: Manipulación de arrays.
* PIL (Pillow): Carga y preprocesamiento de imágenes externas.

