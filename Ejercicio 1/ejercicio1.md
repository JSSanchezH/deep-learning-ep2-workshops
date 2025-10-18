# Clasificación del Dataset Iris con Redes Neuronales (MLPClassifier)

## Descripción

Se entrenaron varios modelos de red neuronal multicapa (MLPClassifier) para clasificar las especies de flores del dataset **Iris**.
Se probaron distintas funciones de activación (`relu`, `tanh`, `logistic`) y diferentes configuraciones de capas ocultas.

---

## Resultados del Entrenamiento

| Activación | Capas        | Accuracy |
| ---------- | ------------ | -------- |
| relu       | (10,)        | 0.9556   |
| relu       | (20, 10)     | 0.9333   |
| relu       | (30, 20, 10) | 0.9333   |
| tanh       | (10,)        | 0.9333   |
| tanh       | (20, 10)     | 0.9333   |
| tanh       | (30, 20, 10) | 0.8889   |
| logistic   | (10,)        | 0.9333   |
| logistic   | (20, 10)     | 0.9333   |
| logistic   | (30, 20, 10) | 0.9111   |

El mejor modelo fue con **activación ReLU** y una sola capa oculta de **10 neuronas**.

---

## Rendimiento del Mejor Modelo

**Modelo:** `MLPClassifier(hidden_layer_sizes=(10,), activation='relu')`

**Precisión global:** 0.9556

**Reporte de clasificación:**
| Clase | Precisión | Recall | F1-score |
|---------------|------------|---------|-----------|
| setosa | 1.00 | 1.00 | 1.00 |
| versicolor | 0.93 | 0.93 | 0.93 |
| virginica | 0.93 | 0.93 | 0.93 |

**Precisión en entrenamiento:** 0.971
**Precisión en prueba:** 0.956

---

## Respuestas

### **A. ¿Qué función de activación funcionó mejor?**

La función **ReLU** obtuvo la mejor precisión (≈0.956).

> **Respuesta:** La mejor función de activación fue **ReLU**.

---

### **B. ¿Cómo afectó el número de capas y neuronas al rendimiento?**

El modelo con **una sola capa de 10 neuronas** tuvo el mejor desempeño.
Aumentar la cantidad de capas y neuronas **no mejoró el accuracy** e incluso lo redujo ligeramente.

> **Respuesta:** Más capas y neuronas no mejoraron el rendimiento; el modelo simple generaliza mejor.

---

### **C. ¿Hay alguna clase que el modelo confunda más que otras?**

Según la matriz de confusión y el reporte, la clase **setosa** se clasificó perfectamente (100%),
mientras que **versicolor** y **virginica** presentaron ligeras confusiones entre sí (recall ≈ 0.93).
Estas dos especies son las más similares en las características del dataset.

> **Respuesta:** El modelo confunde levemente **versicolor** y **virginica**, pero clasifica perfectamente **setosa**.

---

### **D. ¿Se observa Overfitting o Underfitting?**

El modelo presenta:

- **Entrenamiento:** 97.1%
- **Prueba:** 95.6%

La diferencia es pequeña (≈1.5%), por lo que **no hay sobreajuste significativo**.
El modelo generaliza bien y muestra un **ajuste adecuado** a los datos.

> **Respuesta:** No se observa overfitting ni underfitting. El modelo está bien ajustado.
