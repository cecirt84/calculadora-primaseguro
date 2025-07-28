# Calculadora de Prima de Seguro 🚗

**Proyecto final de Bootcamp Data Science (oct-24 – jul-25)**  
Este proyecto nace del análisis de datos de siniestralidad vial (DGT España, 2018–2023) para diseñar una **app** que simula el cálculo de una prima de seguro de coche y su ajuste por riesgo de accidente.

---

## 📝 Resumen del Notebook

1. **Análisis exploratorio**  
   - Dataset de accidentes (DGT).  
   - Preprocesamiento y limpieza de datos.

2. **Cálculo de multiplicadores de riesgo**  
   - Metodología **TOPSIS** para obtener puntuaciones de riesgo por provincia, franja horaria, tipo de zona y tipo de día.  
   - Normalización centrad​a en 1.0 para garantizar que el multiplicador medio sea neutro (±20% rango).

3. **Modelo de prima base**  
   - Creación de un dataset sintético de primas ficticias.  
   - Entrenamiento de un **XGBoost** (`xgb_model.joblib`) para predecir la prima base a partir del perfil del conductor y características del coche.

4. **Despliegue de la app**  
   - Interfaz paso a paso en **Streamlit**:  
     1. Introducción de datos del conductor y vehículo.  
     2. Cálculo de prima base con XGBoost.  
     3. Ajuste de prima vía multiplicador ponderado.  
     4. Resumen con métricas y gráfico de radar.

---

## 🔧 Modelos y Herramientas

- **Método de riesgo**: TOPSIS  
- **Modelo supervisado**: XGBoost (scikit-learn API)  
- **Librerías principales**:  
  - `pandas`, `numpy` – manipulación de datos  
  - `scikit-learn`, `xgboost`, `joblib` – modelado  
  - `streamlit`, `matplotlib`, `seaborn` – interfaz y visualizaciones  

---

## 🌐 Demo en Vivo

Puedes probar la aplicación desplegada en Streamlit Community Cloud:  
https://calculadora-primaseguro.streamlit.app/
