# Modelo de Propensión de Compra para Priorización Comercial - EastWest Airlines

> Proyecto de analítica comercial para estimar la probabilidad de compra de un servicio de telecomunicaciones y priorizar clientes con mayor potencial comercial.

**Autor(a):** Valeria Thais Ureta Llanque  
**Programa:** Diplomado Data Scientist  
**Profesor:** Cezar Quezada (Data Scientist)  
**Nota:** 19  
**Stack:** Python | pandas | numpy | matplotlib | seaborn | scikit-learn | XGBoost

---

## Estructura del proyecto

```text
.
├── input
│   └── EastWestAirlinesNN.csv
├── notebooks
│   ├── EDA.ipynb
│   └── modelos.ipynb
├── output
│   ├── eastwest_model_ready.csv
│   └── eastwest_crm_scoring.csv
├── requirements.txt
└── README.md
```

| Archivo                    | Descripción                                                |
| -------------------------- | ---------------------------------------------------------- |
| `EDA.ipynb`                | Limpieza, validación y análisis exploratorio de la base    |
| `modelos.ipynb`            | Entrenamiento, evaluación de modelos y generación de score |
| `eastwest_model_ready.csv` | Base preparada para modelamiento                           |
| `eastwest_crm_scoring.csv` | Base final con score y priorización de clientes            |

---

## Resumen ejecutivo

EastWest Airlines contaba con una base histórica de clientes y resultados de una campaña comercial asociada a la compra de un servicio de telecomunicaciones.

El objetivo del proyecto fue construir un modelo de propensión de compra que permitiera pasar de una estrategia de contacto masivo a una priorización comercial basada en datos.

A partir del modelo, se generó un score de compra por cliente para ordenar la base y enfocar los esfuerzos comerciales en los clientes con mayor probabilidad de conversión.

---

## Problema de negocio

Una campaña comercial masiva puede asignar el mismo esfuerzo a toda la base de clientes, sin diferenciar entre clientes con alto y bajo potencial de compra.

Esto puede generar:

* Menor eficiencia en los contactos.
* Mayor costo operativo.
* Menor tasa de conversión.
* Pérdida de oportunidades en clientes con mayor potencial.

La pregunta central del proyecto fue:

> **¿A qué clientes debería priorizar el equipo comercial para aumentar la conversión de la campaña?**

---

## Datos utilizados

**Dataset:** `EastWestAirlinesNN.csv`
**Registros analizados:** 4,985 clientes
**Unidad de análisis:** cliente único
**Variable objetivo:** `phone_sale`

La variable `phone_sale` indica si el cliente compró o no el servicio de telecomunicaciones:

* `1` = compró el servicio.
* `0` = no compró el servicio.

La tasa base de compra fue de aproximadamente **20.9%**, es decir, alrededor de 21 de cada 100 clientes compraron el servicio en la base histórica.

---

## Metodología

El proyecto siguió un flujo de trabajo orientado a convertir datos históricos en una herramienta de priorización comercial:

1. Validación y limpieza de la base de clientes.
2. Revisión de la unidad de análisis y variable objetivo.
3. Análisis exploratorio de compradores y no compradores.
4. Entrenamiento de modelos predictivos.
5. Evaluación técnica y comercial de los modelos.
6. Generación de score de compra por cliente.
7. Exportación de una base final priorizada.

---

## Modelos evaluados

Se compararon tres modelos supervisados:

| Modelo              | Propósito                                      |
| ------------------- | ---------------------------------------------- |
| Regresión Logística | Modelo base interpretable                      |
| Random Forest       | Modelo flexible y equilibrado                  |
| XGBoost             | Modelo competitivo para ranking y priorización |

El objetivo no fue únicamente clasificar clientes como compradores o no compradores, sino generar una probabilidad estimada de compra para ordenar la base según potencial comercial.

---

## Resultados técnicos

Debido al desbalance de la variable objetivo, no se evaluó únicamente accuracy. También se revisaron precision, recall, F1-score, ROC-AUC, PR-AUC y Brier score.

| Modelo              | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC | Brier score |
| ------------------- | -------: | --------: | -----: | -------: | ------: | -----: | ----------: |
| Random Forest       |   86.06% |    75.94% | 48.56% |   59.24% |  77.71% | 65.92% |      0.1483 |
| XGBoost             |   81.85% |    57.38% | 50.48% |   53.71% |  77.18% | 65.60% |      0.1474 |
| Regresión Logística |   82.85% |    61.21% | 48.56% |   54.16% |  76.49% | 64.91% |      0.1485 |

**Random Forest** fue el modelo más equilibrado en métricas generales, destacando en accuracy, precision, F1-score, ROC-AUC y PR-AUC.

---

## Resultados de priorización comercial

Además de las métricas técnicas, se evaluó si el modelo ayudaba a priorizar mejor la campaña.

Para ello, se ordenó la base según el score de compra y se calculó la tasa real de compra en los grupos con mayor probabilidad estimada.

La tasa base en el conjunto de prueba fue **20.86%**.

| Modelo              | Compra Top 10% | Mejora Top 10 | Compra Top 20% | Mejora Top 20 | Compra Top 30% | Mejora Top 30 |
| ------------------- | -------------: | ------------: | -------------: | ------------: | -------------: | ------------: |
| Random Forest       |         89.00% |         4.27x |         54.50% |         2.61x |         40.67% |         1.95x |
| XGBoost             |         90.00% |         4.31x |         55.50% |         2.66x |         42.33% |         2.03x |
| Regresión Logística |         92.00% |         4.41x |         52.50% |         2.52x |         40.33% |         1.93x |

El principal hallazgo fue que los clientes con mayor score concentraban una tasa de compra muy superior a la tasa base.

Por ejemplo, con XGBoost, el Top 30% priorizado alcanzó una tasa de compra de **42.33%**, más del doble de la tasa base del conjunto de prueba.

---

## Score de compra

El score de compra corresponde a la probabilidad estimada de que un cliente compre el servicio.

Este score permitió:

* Ordenar clientes de mayor a menor probabilidad de compra.
* Identificar grupos prioritarios para campaña.
* Comparar el desempeño comercial de los modelos.
* Generar una base final utilizable para seguimiento, reportería o selección de audiencias.

---

## Validación de confiabilidad

Durante el análisis se identificó que `club_member`, variable asociada a la pertenencia del cliente al programa de fidelización de la aerolínea, tenía una influencia muy alta en los resultados.

Por ello, se desarrolló un escenario alternativo excluyendo esta variable, con el objetivo de evaluar la dependencia del modelo y revisar un posible riesgo de fuga de información.

| Modelo sin `club_member`              | Accuracy | F1-score | ROC-AUC | PR-AUC | Compra Top 10% | Mejora Top 10 |
| ------------------------------------- | -------: | -------: | ------: | -----: | -------------: | ------------: |
| Regresión Logística sin `club_member` |   58.78% |   37.25% |  63.09% | 32.15% |         42.00% |         2.01x |
| Random Forest sin `club_member`       |   59.38% |   35.82% |  60.79% | 30.22% |         43.00% |         2.06x |
| XGBoost sin `club_member`             |   56.67% |   32.50% |  58.73% | 28.26% |         35.00% |         1.68x |

La caída de desempeño confirmó que esta variable concentraba una parte importante del poder predictivo. Antes de usar el modelo en un contexto real, sería necesario validar si esta información estaba disponible antes de la campaña o si podría representar fuga de información.

---

## Impacto para negocio

El proyecto permite transformar una base histórica de clientes en una herramienta de priorización comercial.

El negocio podría utilizar el score para:

* Enfocar campañas en clientes con mayor probabilidad de conversión.
* Reducir contactos poco eficientes.
* Priorizar recursos comerciales cuando existe capacidad limitada.
* Alimentar campañas piloto, pruebas controladas o listas comerciales priorizadas.
* Dar seguimiento a indicadores de conversión por grupo priorizado.

---

## Conclusiones

* Se validó una base histórica de 4,985 clientes y se confirmó que cada fila representaba un cliente único.
* La tasa base de compra fue aproximadamente 20.9%, utilizada como referencia para evaluar la mejora de priorización.
* Random Forest fue el modelo más equilibrado en métricas generales.
* XGBoost destacó en priorización comercial para los cortes Top 20% y Top 30%.
* El score de compra permitió ordenar clientes y convertir el modelo en un insumo accionable para campañas.
* La revisión de `club_member` permitió identificar una dependencia relevante y reforzar la confiabilidad del análisis.
* El modelo debe usarse como apoyo a la priorización comercial, no como una regla automática de decisión.

---

## Limitaciones

* No se cuenta con eventos detallados de campaña, como envíos, aperturas, clics o rebotes.
* No existe una variable temporal que permita evaluar estabilidad del comportamiento en el tiempo.
* La variable `club_member` debe validarse con negocio antes de una posible implementación.
* El modelo debe probarse con datos futuros antes de considerarse productivo.

---

## Posibles mejoras

* Validar la disponibilidad temporal de variables sensibles antes de la campaña.
* Evaluar el modelo con nuevas campañas o datos futuros.
* Incorporar eventos de contacto y respuesta comercial.
* Construir dashboards de seguimiento con tasa base, tasa por grupo priorizado y conversión.
* Integrar el score en procesos de reportería o priorización operativa.
