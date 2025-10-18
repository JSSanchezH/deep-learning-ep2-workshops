# Predicción de Precios de Vivienda en California con Redes Neuronales (MLPRegressor)

## Descripción

Se entrenaron redes neuronales **MLPRegressor** para predecir el valor medio de las viviendas en el dataset **California Housing**.
Se variaron las **funciones de activación** (`relu`, `tanh`, `logistic`) y las **arquitecturas** de red (número de capas y neuronas).

---

## ⚙️ Resultados del Entrenamiento

| Activación | Capas          | R²     | MAE    | RMSE   |
| ---------- | -------------- | ------ | ------ | ------ |
| relu       | (50,)          | 0.7722 | 0.3759 | 0.5468 |
| relu       | (100, 50)      | 0.7970 | 0.3472 | 0.5162 |
| relu       | (100, 100, 50) | 0.7786 | 0.3533 | 0.5390 |
| tanh       | (50,)          | 0.7682 | 0.3836 | 0.5515 |
| tanh       | (100, 50)      | 0.7998 | 0.3597 | 0.5126 |
| tanh       | (100, 100, 50) | 0.7401 | 0.3967 | 0.5841 |
| logistic   | (50,)          | 0.7705 | 0.3844 | 0.5489 |
| logistic   | (100, 50)      | 0.7883 | 0.3644 | 0.5271 |
| logistic   | (100, 100, 50) | 0.7558 | 0.4184 | 0.5661 |

---

## Mejor Modelo

El mejor desempeño se obtuvo con la configuración:

- **Función de activación:** `tanh`
- **Capas ocultas:** `(100, 50)`
- **R²:** **0.7998**
- **MAE:** **0.3597**
- **RMSE:** **0.5126**

---

## 🧩 Respuestas a las Preguntas

### **A. ¿Qué configuración del modelo dio mejores resultados?**

El modelo con **activación `tanh`** y arquitectura **(100, 50)** alcanzó el mayor R² (≈0.80) y el menor RMSE (≈0.51).
La red con **ReLU (100, 50)** también tuvo un rendimiento muy cercano, por lo que ambas configuraciones son adecuadas.

> **Respuesta:** La mejor configuración fue **(100, 50)** con función de activación **tanh** (R² ≈ 0.80).

---

### **B. ¿Qué importancia tiene la normalización de los datos en este modelo?**

La normalización es **fundamental** en redes neuronales, ya que:

- Permite que los gradientes se propaguen de manera estable durante el entrenamiento.
- Evita que características con valores grandes dominen a las demás.
- Mejora la convergencia y la precisión del modelo.

En este experimento, al entrenar **sin normalizar los datos**, el rendimiento **cayó notablemente**:

| Condición         | R²     | MAE    |
| ----------------- | ------ | ------ |
| Con normalización | 0.7998 | 0.3597 |
| Sin normalización | 0.587  | 0.539  |

Esto muestra que, sin escalar los datos, la red no logra converger correctamente y comete errores mucho mayores.

> **Respuesta:** La normalización es esencial para que el modelo converja y aprenda correctamente.
> Sin normalización, el rendimiento cae considerablemente (R² bajó a 0.59 y el error promedio aumentó a 0.54).

---

### **C. ¿Cuál es el error promedio en la predicción del precio?**

El **error absoluto medio (MAE)** fue de aproximadamente **0.36**, lo que significa que, en promedio, el modelo se equivoca en **0.36 unidades de valor medio de vivienda** (miles de dólares en la escala del dataset).
El **RMSE ≈ 0.51** refuerza que los errores grandes son poco frecuentes.

> **Respuesta:** El error promedio en la predicción del precio es **≈ 0.36 (MAE)** y el RMSE es **≈ 0.51**.

---
