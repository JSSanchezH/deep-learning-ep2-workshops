# Clasificación de Imágenes con LeNet-5 (TensorFlow/Keras)

---

## Descripción del Proyecto

Este proyecto implementa una red neuronal **LeNet-5** utilizando
**TensorFlow y Keras** para la clasificación de imágenes.\
El conjunto de datos fue dividido manualmente en carpetas `train/` y
`test/`, y se aplicaron técnicas de **aumentación de datos** para
mejorar la capacidad de generalización del modelo.

---

## 1. Carga y Preprocesamiento de Datos

- Se utiliza `image_dataset_from_directory()` para cargar las imágenes
  desde carpetas.
- Normalización con `layers.Rescaling(1./255)` para llevar los valores
  de píxeles al rango \[0,1\].
- Aumentación de datos mediante rotaciones, flips y zooms aleatorios.

```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom((0.1, 0.2)),
])
```

Luego se concatenan los datos originales con los aumentados:

```python
augmented_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
train_dataset = train_dataset.concatenate(augmented_dataset)
```

---

## 2️. Arquitectura del Modelo (LeNet-5)

El modelo se compone de varias capas convolucionales y densas, siguiendo
la arquitectura clásica de **LeNet-5**, adaptada a imágenes RGB.

```python
model = models.Sequential([
    layers.Conv2D(16, (5, 5), activation='relu', input_shape=(64,64, 3), padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(32, (5, 5), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])
```

---

## 3️. Entrenamiento

Se entrena el modelo con `Adam` y `sparse_categorical_crossentropy`
durante **40 épocas**.

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_dataset, validation_data=test_dataset, epochs=40)
```

---

## 4️. Evaluación

Se grafican las curvas de **precisión** y **pérdida** tanto para
entrenamiento como validación, y se obtiene el puntaje final sobre el
conjunto de test.

```python
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print(f"Precisión en test: {test_acc*100:.2f}%")
```

---

## 5️. Predicciones

Se visualizan algunas predicciones del modelo comparando etiquetas
reales y predichas, coloreando los títulos en verde o rojo según
acierto.

---

## Resultados Esperados

- El modelo logra una **precisión de entre 70% y 80%**.
- La aumentación de datos mejora la generalización y reduce el
  sobreajuste.

---

## Dependencias

- TensorFlow
- Keras
- Matplotlib
- NumPy

Instalación rápida:

```bash
pip install tensorflow keras matplotlib numpy
```
