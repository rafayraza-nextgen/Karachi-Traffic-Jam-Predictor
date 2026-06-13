# Karachi Traffic Jam Predictor 🚦

## Description
Karachi Traffic Jam Predictor is a Machine Learning project that predicts traffic congestion levels (Low, Medium, or High) in different areas of Karachi. The project uses a Random Forest Classifier and analyzes factors such as area, day, time of day, weather conditions, and holiday status to estimate traffic conditions.

---

## Project Objectives
- Predict traffic congestion levels.
- Analyze traffic patterns in Karachi.
- Visualize traffic trends using charts and heatmaps.
- Demonstrate a complete machine learning workflow.

---

## Features
- Synthetic dataset generation for Karachi traffic.
- Data cleaning and preprocessing.
- Missing value handling.
- Duplicate removal.
- Label encoding of categorical data.
- Data visualization with charts and heatmaps.
- Random Forest model training.
- Model evaluation using accuracy, classification report, and confusion matrix.
- Traffic congestion prediction for new inputs.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

## Dataset Attributes

| Attribute | Description |
|------------|------------|
| Area | Location in Karachi |
| Day | Day of the week |
| Time of Day | Morning, Afternoon, Evening, Night |
| Weather | Clear, Rainy, Foggy, Hot |
| Is Holiday | Yes or No |
| Congestion Level | Low, Medium, High (Target Variable) |

---

## Areas Covered
- Saddar
- Gulshan
- DHA
- Clifton
- Korangi
- Nazimabad
- Malir
- North Karachi

---

## Workflow

### 1. Data Generation
A synthetic dataset of 500 traffic records is generated using random values.

### 2. Data Preprocessing
- Handle missing values
- Remove duplicate records
- Encode categorical variables

### 3. Data Visualization
The following visualizations are generated:
- Congestion Level Distribution
- Congestion by Time of Day
- Congestion by Weather
- High Congestion Heatmap (Area vs Time)

### 4. Model Training
The project uses:

**Random Forest Classifier**
- Number of Trees: 100
- Random State: 42

### 5. Model Evaluation
Evaluation metrics include:
- Accuracy Score
- Classification Report
- Confusion Matrix

### 6. Prediction
The model predicts congestion levels for custom traffic scenarios.

---

## Output Files
The program generates:

- `traffic_analysis.png`
- `confusion_matrix.png`
- `feature_importance.png`

---

## Sample Prediction

Input:

```python
('Saddar', 'Monday', 'Morning', 'Rainy', 'No')
```

Output:

```text
Predicted Congestion: High
```

---

## Key Findings
- Morning and Evening hours have the highest congestion.
- Rainy and Foggy weather increase traffic congestion.
- Saddar and Clifton are among the most congested areas.
- Time of Day and Weather are the most influential features.

---

## Installation

Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Run the Project

```bash
python traffic_predictor.py
```

---

## Future Enhancements
- Real-time traffic data integration.
- Google Maps API integration.
- Web-based dashboard.
- Deep learning-based prediction models.
- Live traffic forecasting system.

---

## Author

**Rafay Raza**  
BSAI – Dawood University of Engineering & Technology

---

## License

This project is developed for educational and academic purposes only.
