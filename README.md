# Predicción de Compra del Servicio Telcom - EastWest Airlines

> Modelo supervisado para predecir la probabilidad de compra del servicio Telcom y priorizar clientes con mayor potencial comercial para campañas de marketing.

**Autor(a):** Valeria Thais Ureta Llanque    
**Programa:** Diplomado Data Scientist  
**Profesor:** Cezar Quezada (Data Scientist)  
**Nota:** 19  
**Stack:** Python | pandas | numpy | matplotlib | seaborn | scikit-learn | XGBoost  

---

# Problema identificado

EastWest Airlines busca aumentar la efectividad comercial de sus campañas para el servicio Telcom, priorizando a los clientes con mayor probabilidad de compra y reduciendo contactos poco rentables.

Actualmente, una campaña masiva puede asignar el mismo esfuerzo comercial a toda la base de clientes, sin diferenciar entre clientes con alta propensión de compra, clientes con señales débiles o clientes que requieren una estrategia distinta de comunicación. Esto puede generar mayores costos de contacto, menor tasa de conversión y pérdida de oportunidades comerciales en segmentos con mayor potencial.

El objetivo del proyecto es construir un modelo supervisado que permita estimar la probabilidad de compra del servicio Telcom por cliente. Con esta información, se puede ordenar la base comercial, priorizar los contactos más prometedores y diseñar campañas más eficientes, enfocadas en conversión y optimización de recursos.

**Dataset:** `EastWestAirlinesNN.csv`  
**Granularidad del dataset:** Cada registro representa un cliente único identificado por `ID#`, con características de fidelización, acumulación de millas, actividad reciente y compra del servicio Telcom.

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
│   └── eastwest_model_ready.csv               # Base preparada para modelamiento
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
10. Experimento de sensibilidad excluyendo `club_member`.
11. Conclusiones finales para negocio.

---

## Modelos utilizados

Se compararon tres modelos supervisados:

- **Regresión Logística:** modelo base interpretable.
- **Random Forest:** modelo flexible para capturar relaciones no lineales.
- **XGBoost:** modelo orientado a mejorar el ranking comercial.

---

## Métricas principales

| Modelo | Accuracy | F1-score | ROC-AUC | PR-AUC | Top 10% compra | Lift Top 10 |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 86.06% | 59.24% | 77.71% | 65.92% | 89.00% | 4.27 |
| XGBoost | 81.85% | 53.71% | 77.18% | 65.60% | 90.00% | 4.31 |
| Regresión Logística | 82.85% | 54.16% | 76.49% | 64.91% | 92.00% | 4.41 |

Random Forest fue el modelo más equilibrado en métricas generales.  
XGBoost mostró buen desempeño cuando el objetivo es maximizar la priorización comercial.

---

## Experimento sin `club_member`

Durante el análisis se detectó que la variable `club_member` tenía una relación extremadamente fuerte con la compra del servicio Telcom.

Por este motivo, se entrenó una segunda versión de los modelos excluyendo esta variable, con el objetivo de evaluar una posible fuga de información.

| Modelo sin `club_member` | Accuracy | F1-score | ROC-AUC | PR-AUC | Top 10% compra | Lift Top 10 |
|---|---:|---:|---:|---:|---:|---:|
| Regresión Logística sin `club_member` | 58.78% | 37.25% | 63.09% | 32.15% | 42.00% | 2.01 |
| Random Forest sin `club_member` | 59.38% | 35.82% | 60.79% | 30.22% | 43.00% | 2.06 |
| XGBoost sin `club_member` | 56.67% | 32.50% | 58.73% | 28.26% | 35.00% | 1.68 |

La caída de desempeño confirma que `club_member` es una variable crítica. Antes de usar el modelo en producción, debe validarse si esta variable estaba disponible antes de la campaña o si podría representar fuga de información.

---

## Conclusiones

- El modelo permite pasar de una campaña masiva a una estrategia de contacto priorizada, ordenando a los clientes según su probabilidad estimada de compra del servicio Telcom.

- En el escenario con `club_member`, los modelos logran una fuerte concentración de compradores en los primeros cortes de priorización. En el Top 10%, la tasa de compra estimada se ubica entre 89% y 92%, frente a una tasa base cercana al 21%.

- Random Forest es el modelo más recomendable si el negocio busca un balance entre precisión, estabilidad e interpretación operativa. Alcanzó el mejor ROC-AUC y el mejor F1-score entre los modelos evaluados.

- XGBoost es una alternativa relevante si la prioridad comercial es maximizar el ranking de clientes en campañas con capacidad limitada, ya que obtuvo el mejor desempeño en el Top 30% priorizado.

- La variable `club_member` concentra gran parte del poder predictivo del modelo. Antes de usarla en una campaña real, debe confirmarse que estaba disponible antes del evento de compra y que no representa fuga de información.

- Si `club_member` no puede validarse, el escenario sin esta variable debe tomarse como referencia más conservadora. En ese caso, el desempeño disminuye de forma importante, pero aún permite mejorar la priorización respecto a una selección no segmentada.

- La recomendación de negocio es usar el modelo como herramienta de priorización comercial, no como regla automática de decisión. Los clientes con mayor score deberían alimentar campañas piloto, pruebas A/B o listas comerciales priorizadas.

---

## Limitaciones

- La variable `club_member` debe validarse con negocio para confirmar que estaba disponible antes de la campaña y descartar fuga de información.
- No se cuenta con una variable temporal de campaña, por lo que no es posible evaluar estabilidad del comportamiento en el tiempo.
- El modelo debe probarse con datos futuros antes de usarse en producción.
- Las recomendaciones deben validarse mediante campañas piloto o pruebas A/B.
