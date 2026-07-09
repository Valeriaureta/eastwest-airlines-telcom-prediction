# Modelo de Propensión de Compra para Priorización Comercial - EastWest Airlines

> Proyecto de analítica de datos y modelamiento predictivo para estimar la probabilidad de compra de un servicio de telecomunicaciones y priorizar clientes con mayor potencial comercial.

**Autor(a):** Valeria Thais Ureta Llanque  
**Programa:** Diplomado Data Scientist  
**Profesor:** Cezar Quezada (Data Scientist)  
**Nota:** 19  
**Stack:** Python | pandas | numpy | matplotlib | seaborn | scikit-learn | XGBoost

---

# Problema identificado

EastWest Airlines contaba con una base histórica de clientes y resultados de una campaña comercial asociada a la compra de un servicio de telecomunicaciones. El reto consistía en analizar esta información para identificar qué clientes tenían mayor probabilidad de compra y cómo priorizarlos de forma más eficiente.

Una estrategia comercial masiva puede asignar el mismo esfuerzo a toda la base de clientes, sin diferenciar entre clientes con alto potencial de conversión y clientes con menor probabilidad de respuesta. Esto puede generar menor eficiencia en los contactos, mayores costos operativos y pérdida de oportunidades en clientes con mayor potencial.

El objetivo del proyecto fue construir un modelo supervisado que permita estimar la probabilidad de compra por cliente. A partir de esa probabilidad, se generó un score de compra para ordenar la base y facilitar la priorización de audiencias para futuras acciones comerciales.

**Dataset:** `EastWestAirlinesNN.csv`
**Granularidad del dataset:** Cada registro representa un cliente único identificado por `ID#`, con variables asociadas a fidelización, acumulación de millas, actividad reciente, compras online y compra del servicio de telecomunicaciones.

---

## Estructura del proyecto

```text
.
├── input
│   └── EastWestAirlinesNN.csv                 # Dataset original
├── notebooks
│   ├── EDA.ipynb                              # Limpieza, validación y análisis exploratorio
│   └── modelos.ipynb                          # Modelamiento predictivo y comparación de modelos
├── output
│   ├── eastwest_model_ready.csv               # Base preparada para modelamiento
│   └── eastwest_crm_scoring.csv               # Base final con score y priorización de clientes
├── requirements.txt
└── README.md
```

---

# Notebooks principales

## `notebooks/EDA.ipynb`

El notebook contiene:

1. Entendimiento inicial del dataset.
2. Estandarización de nombres de columnas.
3. Revisión de granularidad y unidad de análisis.
4. Corrección de tipos de datos.
5. Validación de variables binarias.
6. Tratamiento de valores nulos.
7. Revisión de duplicados.
8. Análisis de outliers.
9. Análisis univariado y bivariado.
10. Revisión de la variable objetivo `phone_sale`.
11. Exportación de la base preparada para modelamiento.

---

## `notebooks/modelos.ipynb`

El notebook contiene:

1. Carga de la base preparada.
2. Definición de variables predictoras y variable objetivo.
3. Revisión del desbalance de clases.
4. Split train-test estratificado.
5. Entrenamiento de Regresión Logística.
6. Entrenamiento de Random Forest.
7. Entrenamiento de XGBoost.
8. Comparación de modelos con métricas predictivas y comerciales.
9. Evaluación de priorización en Top 10%, Top 20% y Top 30%.
10. Generación de score de compra por cliente.
11. Experimento de sensibilidad excluyendo `club_member`.
12. Conclusiones finales para negocio.

---

# Preparación y validación de datos

Antes del modelamiento se realizó una revisión de calidad de datos para asegurar que la base fuera consistente y utilizable para análisis.

Las principales actividades fueron:

* Validación de nombres de columnas, tipos de datos y variables binarias.
* Revisión de valores nulos y duplicados.
* Confirmación de la unidad de análisis: cada fila representa un cliente único.
* Análisis de outliers en variables como millas, vuelos y transacciones.
* Preparación de una base final para modelamiento: `eastwest_model_ready.csv`.

Se eliminaron registros sin identificador de cliente ni resultado de compra, ya que no permitían asociar correctamente la información a un cliente ni validar la variable objetivo.

---

# Variable objetivo

La variable objetivo fue:

```text
phone_sale
```

Donde:

* `1` = el cliente compró el servicio de telecomunicaciones.
* `0` = el cliente no compró el servicio.

En la base histórica, 1,042 de 4,985 clientes compraron el servicio, lo que representa una tasa base de compra de aproximadamente 20.9%.

Esta tasa fue utilizada como punto de comparación para evaluar si los modelos lograban identificar grupos de clientes con mayor probabilidad de conversión.

---

# Modelos utilizados

Se compararon tres modelos supervisados:

* **Regresión Logística:** modelo base interpretable.
* **Random Forest:** modelo flexible para capturar relaciones no lineales.
* **XGBoost:** modelo competitivo para ranking y priorización comercial.

El objetivo no fue únicamente clasificar clientes como compradores o no compradores, sino generar una probabilidad estimada de compra que permita ordenar la base según potencial comercial.

---

# Métricas principales

Debido a que la base estaba desbalanceada, no se utilizó únicamente accuracy. También se evaluaron métricas como precision, recall, F1-score, ROC-AUC, PR-AUC y Brier score.

| Modelo              | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC | Brier score |
| ------------------- | -------: | --------: | -----: | -------: | ------: | -----: | ----------: |
| Random Forest       |   86.06% |    75.94% | 48.56% |   59.24% |  77.71% | 65.92% |      0.1483 |
| XGBoost             |   81.85% |    57.38% | 50.48% |   53.71% |  77.18% | 65.60% |      0.1474 |
| Regresión Logística |   82.85% |    61.21% | 48.56% |   54.16% |  76.49% | 64.91% |      0.1485 |

Random Forest fue el modelo más equilibrado en métricas generales, destacando en accuracy, precision, F1-score, ROC-AUC y PR-AUC.

---

# Evaluación de priorización comercial

Además de las métricas predictivas, se evaluó la mejora de priorización comercial. Para ello, los clientes fueron ordenados según su score de compra y se calculó la tasa real de compra en los grupos con mayor probabilidad estimada.

La tasa base en el conjunto de prueba fue 20.86%.

| Modelo              | Tasa base test | Compra Top 10% | Lift Top 10% | Compra Top 20% | Lift Top 20% | Compra Top 30% | Lift Top 30% |
| ------------------- | -------------: | -------------: | -----------: | -------------: | -----------: | -------------: | -----------: |
| Random Forest       |         20.86% |         89.00% |         4.27 |         54.50% |         2.61 |         40.67% |         1.95 |
| XGBoost             |         20.86% |         90.00% |         4.31 |         55.50% |         2.66 |         42.33% |         2.03 |
| Regresión Logística |         20.86% |         92.00% |         4.41 |         52.50% |         2.52 |         40.33% |         1.93 |

La evaluación mostró que los clientes con mayor score concentraban una tasa de compra significativamente superior a la tasa base. Por ejemplo, en el Top 30% priorizado por XGBoost, la tasa de compra fue 42.33%, más del doble de la tasa base del conjunto de prueba.

Esto evidencia que el modelo puede aportar valor como herramienta de priorización comercial.

---

# Generación de score de compra

Después de comparar los modelos, se generó un score de compra por cliente. Este score corresponde a la probabilidad estimada de compra del servicio de telecomunicaciones.

La lógica fue:

1. Entrenar y evaluar los modelos.
2. Seleccionar un modelo de referencia.
3. Calcular la probabilidad estimada de compra por cliente.
4. Crear una columna de score de compra.
5. Ordenar la base de clientes según el score.
6. Definir grupos priorizados para futuras acciones comerciales.

La salida final fue una base estructurada con información como:

* ID del cliente.
* Score de compra.
* Grupo de priorización.
* Variables relevantes del cliente.
* Resultado real de compra.

Esta base puede ser utilizada como insumo para seguimiento comercial, reportería o selección de audiencias.

---

# Experimento sin `club_member`

Durante el análisis se detectó que la variable `club_member` tenía una influencia muy alta en los resultados del modelo. Esta variable representa la pertenencia del cliente al programa de fidelización de la aerolínea.

Por este motivo, se entrenó una segunda versión de los modelos excluyendo esta variable, con el objetivo de evaluar la dependencia del modelo frente a este atributo y revisar un posible riesgo de fuga de información.

| Modelo sin `club_member`              | Accuracy | F1-score | ROC-AUC | PR-AUC | Top 10% compra | Lift Top 10 |
| ------------------------------------- | -------: | -------: | ------: | -----: | -------------: | ----------: |
| Regresión Logística sin `club_member` |   58.78% |   37.25% |  63.09% | 32.15% |         42.00% |        2.01 |
| Random Forest sin `club_member`       |   59.38% |   35.82% |  60.79% | 30.22% |         43.00% |        2.06 |
| XGBoost sin `club_member`             |   56.67% |   32.50% |  58.73% | 28.26% |         35.00% |        1.68 |

La caída de desempeño confirmó que esta variable concentraba una parte importante del poder predictivo. Antes de utilizar el modelo en un entorno real, sería necesario validar con negocio si esta información estaba disponible antes de la campaña o si podría representar fuga de información.

---

# Conclusiones

* Se gestionó y validó una base histórica de 4,985 clientes, confirmando la unidad de análisis y preparando la información para modelamiento.

* La tasa base de compra fue aproximadamente 20.9%, lo que permitió establecer un punto de comparación para evaluar la mejora generada por los modelos.

* Random Forest fue el modelo más equilibrado en métricas generales, destacando en precision, F1-score, ROC-AUC y PR-AUC.

* XGBoost fue una alternativa competitiva para priorización comercial, especialmente en los cortes Top 20% y Top 30%.

* La generación del score de compra permitió transformar el resultado del modelo en una base ordenada para priorizar clientes con mayor potencial.

* El análisis de sensibilidad sin `club_member` permitió identificar una dependencia relevante del modelo y reforzar la necesidad de validar posibles riesgos de fuga de información antes de una implementación real.

* La recomendación es utilizar el modelo como herramienta de apoyo para priorización comercial, no como una regla automática de decisión. Los clientes con mayor score pueden alimentar campañas piloto, pruebas controladas o listas comerciales priorizadas.

---

# Limitaciones

* La variable `club_member` debe validarse con negocio para confirmar que estaba disponible antes de la campaña y descartar fuga de información.
* No se cuenta con una variable temporal de campaña, por lo que no es posible evaluar estabilidad del comportamiento en el tiempo.
* No se dispone de eventos detallados de campaña, como envíos, aperturas, clics o rebotes.
* El modelo debe probarse con datos futuros antes de considerarse para una implementación productiva.

---

# Posibles mejoras

* Validar la disponibilidad temporal de variables sensibles antes de la campaña.
* Evaluar el modelo con datos futuros o nuevas campañas.
* Incorporar eventos de contacto o respuesta comercial si estuvieran disponibles.
* Construir dashboards de seguimiento para monitorear tasa base, tasas por grupo priorizado y desempeño de campañas.
* Integrar el score en procesos operativos de priorización comercial o reportería.
