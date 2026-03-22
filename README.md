# 🌍 ClimateScope AI: Enterprise Climate Intelligence Platform
**An Advanced Unsupervised Machine Learning Engine for Global Risk Analysis**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud_Deployed-FF4B4B.svg)
![ML](https://img.shields.io/badge/Machine_Learning-K--Means_%7C_PCA-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> **Lead Architect:** GURU  
> **Deployment Status:** Live on Streamlit Cloud

---

## 📖 Project Overview
**ClimateScope** is a high-performance SaaS-style dashboard designed to bridge the gap between raw meteorological big data and actionable systemic risk insights. By processing over **120,000+ localized data points**, the platform exposes hidden compounding risks of global climate shifts through real-time mathematical modeling and interactive 3D visualizations.

## 🧠 Platform Intelligence Modules

### 1. 📊 Executive Insights & Correlation Engine
* **Statistical Analysis:** Automatically calculates Pearson correlation coefficients between key climate variables (Temperature, Humidity, Wind Speed).
* **Dynamic Trending:** Dual-axis visualization comparing precipitation patterns and temperature anomalies over custom-selected timeframes.

### 2. 🌐 Global Climate Risk Mapping
* **Animated 3D Geospatial Engine:** A chronological 3D globe mapping a custom-calculated **Systemic Risk Score** across 200+ nations.
* **Temporal Evolution:** Users can "play" the timeline to watch how climate risk has migrated geographically over the last decade.

### 3. 🤖 AI Pattern Detection (Machine Learning)
* **Unsupervised Clustering:** Utilizes **K-Means Clustering** to categorize nations into "Climate Archetypes" based on 5D feature sets.
* **Dimensionality Reduction (PCA):** Projects complex climate data into a **3D Principal Component Analysis (PCA)** space, allowing users to see how mathematically similar different regions are.
* **Radar Profiling:** Dynamic radar charts break down the specific "DNA" of each AI-generated cluster.

### 4. 📈 Seasonal Time-Series Decomposition
* **Signal Processing:** Uses additive/multiplicative decomposition to strip away daily weather "noise."
* **Macro-Trend Extraction:** Isolate the underlying trend line from seasonal cycles and chaotic residual (extreme) events.

### 5. 🔬 "What-If" Scenario Simulator
* **Interactive Forecasting:** A distribution-based simulator that allows users to inject theoretical global temperature increases (e.g., +1.5°C or +2.0°C).
* **Risk Probability Shifting:** Visualizes how shifting averages significantly increase the frequency of "Tail-End" extreme weather events.

### 6. 🌪️ Extreme Event Analytics
* **Anomaly Detection:** Specifically isolates the absolute boundaries of global weather data.
* **Custom Filtering:** Allows users to distinguish between localized extreme spikes and sustained month-long anomalies.

---

## ⚙️ Technical Architecture
The platform is built on a modern "Data-First" stack designed for low latency and high interactivity:

* **Language:** Python 3.10+
* **Visual Engine:** Plotly Enterprise (3D & Animated Graphs)
* **ML Stack:** Scikit-Learn (`KMeans`, `PCA`, `StandardScaler`)
* **Stats Stack:** Statsmodels (`seasonal_decompose`), Pandas, NumPy
* **UI/UX:** Streamlit with custom **Glassmorphism CSS** (Deep Dark Mode)

---

## 💻 Local Setup & Deployment

1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/springboardmentor3010s/ClimateScope.git](https://github.com/springboardmentor3010s/ClimateScope.git)
   cd ClimateScope