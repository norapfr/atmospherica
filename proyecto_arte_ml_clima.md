# 🎨🌦️ Proyecto: Generador de Pintura Climática con ML

## 📌 Descripción general
Este proyecto combina **arte generativo + machine learning + datos climáticos reales** para crear **cuadros abstractos** que representan el estado atmosférico y, además, incorporan **capacidad predictiva**.

No es una visualización tradicional: defines un **lenguaje visual propio** donde cada elemento artístico codifica información climática.

---

## 🧩 Arquitectura del proyecto

El sistema se divide en 3 módulos independientes:

### 1. Generador de pintura (Semanas 1-3)
- Llamadas a API (OpenWeatherMap)
- Normalización de datos
- Mapeo a parámetros visuales

#### Gramática visual base:
- Temperatura → color (frío = azul oscuro, calor = ámbar)
- Presión → densidad de capas
- PM2.5 → fragmentación
- Viento → dirección de trazos
- Humedad → opacidad

📌 Output: imagen tipo “cuadro abstracto”

---

### 2. Modelo predictivo (Semanas 4-7)
Objetivo: predecir eventos extremos (calor, contaminación, tormentas)

#### Modelos:
- Inicial: Random Forest (scikit-learn)
- Avanzado: LSTM (PyTorch/Keras)

#### Features:
- Medias móviles (7 días)
- Gradientes (cambios diarios)
- Estacionalidad (sin/cos)

#### Datos:
- ERA5 (Copernicus)
- AEMET (España)

📌 Output: predicción que influye en el arte generado

---

### 3. Automatización y archivo (Semanas 8-10)
- Script diario (cron o GitHub Actions)
- Generación automática de cuadros
- Publicación en web (GitHub Pages)

📌 Resultado: sistema vivo que genera arte cada día

---

## 🛠️ Stack técnico

- Datos: requests, pandas, cdsapi
- Arte: cairo / Pillow / p5.js
- ML: scikit-learn, PyTorch
- Infra: GitHub Actions
- Web: HTML/CSS básico

---

## 🚀 Cómo empezar (primeros 3 días)

### Día 1
- Obtener API key de OpenWeatherMap
- Hacer primera llamada y explorar JSON

### Día 2
- Función: datos → parámetros visuales

### Día 3
- Generar primer cuadro simple

---

## 🎯 Objetivo del proyecto

Crear un sistema que:
- Ingiere datos reales
- Aprende patrones climáticos
- Genera arte con significado
- Funciona automáticamente en producción

---

## 💼 Valor para portfolio

Demuestra:
- Ingeniería de datos
- ML aplicado (series temporales)
- Diseño de sistemas
- Arte generativo con semántica

---

## 🧠 Cómo explicarlo en entrevista

“Sistema de ML que ingiere datos climáticos en tiempo real, predice eventos extremos mediante modelos de series temporales y genera representaciones artísticas basadas en una gramática visual que codifica información atmosférica.”

---

## 📌 Clave diferencial

No es:
❌ Arte bonito  
❌ Dashboard  

Es:
✅ Lenguaje visual de datos  
✅ Sistema autónomo  
✅ ML + arte con propósito  

