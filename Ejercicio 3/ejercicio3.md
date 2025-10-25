# Clasificación de Imágenes con CNN (LeNet-5) - Fashion MNIST

## Descripción del Proyecto

En este ejercicio se implementa una **Red Neuronal Convolucional (CNN)**
inspirada en la arquitectura **LeNet-5**, para clasificar imágenes del
conjunto de datos **Fashion MNIST**, que contiene prendas de ropa en
escala de grises de 28x28 píxeles.

---

## 1. Carga y Visualización de Datos

Se utiliza el dataset `fashion_mnist` de Keras, que se divide en 60,000
imágenes para entrenamiento y 10,000 para prueba.\
Cada imagen es normalizada (valores entre 0 y 1) y se ajusta su
dimensión a `(28, 28, 1)` para ser compatible con una CNN.

Ejemplo de etiquetas utilizadas:

Índice Clase

---

0 Camiseta
1 Pantalón
2 Buzo
3 Vestido
4 Abrigo
5 Sandalia
6 Camisa
7 Tenis
8 Bolso
9 Bota

Se visualizan 10 imágenes aleatorias del conjunto de entrenamiento.

---

## 2. Construcción del Modelo (LeNet-5 Modificada)

Se crea un modelo secuencial inspirado en **LeNet-5**, pero modernizado
con funciones de activación ReLU y MaxPooling:

```python
model = models.Sequential([
    layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1), padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(16, (5, 5), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

El modelo se compila con: - Optimizador: **Adam** - Pérdida: **Sparse
Categorical Crossentropy** - Métrica: **Precisión (Accuracy)**

---

## 3. Entrenamiento del Modelo

Se entrena el modelo con los siguientes parámetros:

Parámetro Valor

---

Épocas 20
Batch size 128
Validación Conjunto de prueba (x_test, y_test)

Durante el entrenamiento, se monitorea la precisión y la pérdida para
entrenamiento y validación.

### Gráfica de Precisión

La gráfica muestra cómo evoluciona la precisión y la perdida durante las 20 épocas.\
Generalmente se observa un aumento/disminución constante en ambas curvas hasta
estabilizarse, indicando una buena convergencia.

---

## 4. Evaluación del Modelo

El modelo se evalúa con el conjunto de prueba:

```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Precisión en test: {test_acc*100:.2f}%")
```

**Resultado obtenido:**\
\> Precisión en test: **90%**.

---

## 5. Predicciones

Se generan predicciones sobre las imágenes del conjunto de prueba.\
Para cada imagen se compara la clase predicha con la real:

- **Verde:** predicción correcta
- **Rojo:** predicción incorrecta

Esto permite visualizar de manera cualitativa el desempeño del modelo.

---

## Dependencias

- TensorFlow / Keras\
- Matplotlib\
- NumPy

Instalación rápida:

```bash
pip install tensorflow matplotlib numpy
```

---
