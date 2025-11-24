# 🔧 Sensor Fault Detection with XGBoost

This project implements an end-to-end machine learning pipeline to detect faulty sensors using a binary XGBoost classifier. It includes exploratory data analysis, data cleaning, preprocessing, model training, evaluation, and threshold optimization to improve real-world fault detection.

## 📘 About This Project
This notebook builds a complete workflow for identifying sensor failures using supervised learning. It covers EDA, encoding fixes, meaningful outlier analysis, categorical encoding, handling class imbalance with AUC-PR, and training an XGBoost model.  
Optimizing the decision threshold significantly boosts recall and F1-score, reducing undetected faulty sensors.

## 📊 About the Dataset
The dataset contains measurements related to sensor behavior and operating conditions:

- **humedad_media** — Average humidity (24h)  
- **desviacion_humedad** — Humidity variation  
- **lecturas_anormales** — Out-of-range reading count  
- **alertas_historicas** — Number of past alerts  
- **horas_ultimo_mto** — Hours since last maintenance  
- **uso_sensor** — Estimated usage percentage  
- **interaccion_termica** — Derived technical variable  
- **entorno_fisico** — Physical environment  
- **estado_sensor** — Target variable (0 = normal, 1 = failing)

## 🔍 1. Exploratory Analysis & Preprocessing
- Normalized text due to non-UTF8 characters.  
- Checked data types and validated no missing values.  
- Outliers were kept, as they represent real failing sensors.  
- Encoded non-ordinal categorical variables for modeling.

## ⚙️ 2. Model Training
- Identified strong class imbalance (≈95% normal, 5% failing).  
- Applied stratified 80/10/10 data split.  
- Used **AUC-PR** as the main metric.  
- Trained an XGBoost classifier tuned for sensitivity to the minority class.

## 📈 3. Evaluation
Default threshold (0.5) results:  
- **F1-score:** 0.71  
- **AUC-PR:** 0.54  
- **Precision:** 83.3%  
- **Recall:** 62.5%

## 🎯 4. Optimal Threshold
Using the Precision-Recall curve, **0.43** was selected as the optimal threshold.  
New metrics:  
- **F1-score:** 0.87  
- **Precision:** 87.5%  
- **Recall:** 87.5%

This greatly improves the detection of failing sensors while minimizing false negatives.

## 🚀 Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib / Seaborn  

## 📝 Author
Created by **Juan Sebastián Granados Jaimes** as part of a machine learning and model evaluation study.  
Feel free to use this project as inspiration for your own workflow!

⭐ *If you found this useful, consider starring the repo!*  
