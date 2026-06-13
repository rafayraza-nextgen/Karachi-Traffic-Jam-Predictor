# ============================================================
#   LAB #14: COMPLEX COMPUTING ACTIVITY
#   Project: Karachi Traffic Jam Predictor
#   Student: BSAI - Dawood University of Engineering & Technology
#   Roll No: 24F-AI-075
# ============================================================

# ── IMPORTS ─────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# TASK 1: PROJECT DEFINITION
# ============================================================
# Goal    : Predict traffic congestion level (Low/Medium/High)
# Features: Area, Day, Time of Day, Weather, Is Holiday
# Target  : Congestion Level
# Algorithm: Random Forest Classifier
# ============================================================


# ============================================================
# TASK 2: LOAD AND PREPROCESS DATA
# ============================================================

print("=" * 55)
print("   KARACHI TRAFFIC JAM PREDICTOR")
print("=" * 55)
print("\n[1] LOADING DATASET...\n")

# --- Step 1: Create Synthetic Dataset ---
np.random.seed(42)
n = 500

areas       = ['Saddar', 'Gulshan', 'DHA', 'Clifton', 'Korangi',
                'Nazimabad', 'Malir', 'North Karachi']
days        = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
               'Friday', 'Saturday', 'Sunday']
times       = ['Morning', 'Afternoon', 'Evening', 'Night']
weathers    = ['Clear', 'Rainy', 'Foggy', 'Hot']
holidays    = ['Yes', 'No']

data = pd.DataFrame({
    'area'        : np.random.choice(areas,    n),
    'day'         : np.random.choice(days,     n),
    'time_of_day' : np.random.choice(times,    n),
    'weather'     : np.random.choice(weathers, n),
    'is_holiday'  : np.random.choice(holidays, n),
})

# --- Step 2: Generate Realistic Congestion Labels ---
def assign_congestion(row):
    score = 0
    if row['time_of_day'] in ['Morning', 'Evening']:  score += 2
    if row['weather']     in ['Rainy', 'Foggy']:       score += 2
    if row['area']        in ['Saddar', 'Clifton']:    score += 1
    if row['day']         in ['Monday', 'Friday']:     score += 1
    if row['is_holiday']  == 'Yes':                    score -= 2
    if score >= 4:   return 'High'
    elif score >= 2: return 'Medium'
    else:            return 'Low'

data['congestion_level'] = data.apply(assign_congestion, axis=1)

# --- Step 3: Introduce Some Missing Values ---
for col in ['weather', 'is_holiday']:
    missing_idx = np.random.choice(data.index, size=10, replace=False)
    data.loc[missing_idx, col] = np.nan

print("Dataset Shape:", data.shape)
print("\nFirst 5 Rows:")
print(data.head())
print("\nMissing Values Before Cleaning:")
print(data.isnull().sum())

# --- Step 4: Handle Missing Values ---
data['weather']    = data['weather'].fillna('Clear')
data['is_holiday'] = data['is_holiday'].fillna('No')

# --- Step 5: Remove Duplicates ---
data.drop_duplicates(inplace=True)

print("\nMissing Values After Cleaning:")
print(data.isnull().sum())
print("\nDataset after cleaning:", data.shape)

# --- Step 6: Encode Categorical Columns ---
le = LabelEncoder()
encoded_data = data.copy()
cat_cols = ['area', 'day', 'time_of_day', 'weather', 'is_holiday', 'congestion_level']
for col in cat_cols:
    encoded_data[col] = le.fit_transform(encoded_data[col])

print("\nEncoded Dataset (first 5 rows):")
print(encoded_data.head())


# ============================================================
# TASK 3: ANALYZE AND VISUALIZE DATA
# ============================================================

print("\n[2] GENERATING VISUALIZATIONS...\n")

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Karachi Traffic Jam Analysis', fontsize=16, fontweight='bold')

# Chart 1: Congestion Level Distribution
congestion_counts = data['congestion_level'].value_counts()
axes[0, 0].bar(congestion_counts.index, congestion_counts.values,
               color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
axes[0, 0].set_title('Overall Congestion Level Distribution')
axes[0, 0].set_xlabel('Congestion Level')
axes[0, 0].set_ylabel('Count')
for i, v in enumerate(congestion_counts.values):
    axes[0, 0].text(i, v + 3, str(v), ha='center', fontweight='bold')

# Chart 2: Congestion by Time of Day
time_congestion = data.groupby(['time_of_day', 'congestion_level']).size().unstack(fill_value=0)
time_congestion.plot(kind='bar', ax=axes[0, 1],
                     color=['#e74c3c', '#2ecc71', '#f39c12'], edgecolor='black')
axes[0, 1].set_title('Congestion by Time of Day')
axes[0, 1].set_xlabel('Time of Day')
axes[0, 1].set_ylabel('Count')
axes[0, 1].tick_params(axis='x', rotation=0)
axes[0, 1].legend(title='Congestion Level')

# Chart 3: Congestion by Weather
weather_congestion = data.groupby(['weather', 'congestion_level']).size().unstack(fill_value=0)
weather_congestion.plot(kind='bar', ax=axes[1, 0],
                        color=['#e74c3c', '#2ecc71', '#f39c12'], edgecolor='black')
axes[1, 0].set_title('Congestion by Weather Condition')
axes[1, 0].set_xlabel('Weather')
axes[1, 0].set_ylabel('Count')
axes[1, 0].tick_params(axis='x', rotation=0)
axes[1, 0].legend(title='Congestion Level')

# Chart 4: Heatmap - Area vs Time of Day (High Congestion only)
high_only = data[data['congestion_level'] == 'High']
heatmap_data = high_only.groupby(['area', 'time_of_day']).size().unstack(fill_value=0)
sns.heatmap(heatmap_data, ax=axes[1, 1], cmap='YlOrRd',
            annot=True, fmt='d', linewidths=0.5)
axes[1, 1].set_title('High Congestion: Area vs Time of Day')
axes[1, 1].set_xlabel('Time of Day')
axes[1, 1].set_ylabel('Area')

plt.tight_layout()
plt.savefig('traffic_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Charts saved as 'traffic_analysis.png'")


# ============================================================
# TASK 4: BUILD THE MODEL
# ============================================================

print("\n[3] TRAINING THE MODEL...\n")

# --- Features and Target ---
X = encoded_data[['area', 'day', 'time_of_day', 'weather', 'is_holiday']]
y = encoded_data['congestion_level']

# --- Train/Test Split (80/20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# FIX: Actual values after drop_duplicates → 348 train, 88 test
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# --- Train Random Forest ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\nModel trained successfully!")


# ============================================================
# TASK 5: EVALUATE THE MODEL
# ============================================================

print("\n[4] EVALUATING THE MODEL...\n")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy : {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['High', 'Low', 'Medium']))

# --- Confusion Matrix Plot ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['High', 'Low', 'Medium'],
            yticklabels=['High', 'Low', 'Medium'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix saved as 'confusion_matrix.png'")

# --- Feature Importance ---
importances = model.feature_importances_
feature_names = ['Area', 'Day', 'Time of Day', 'Weather', 'Is Holiday']

plt.figure(figsize=(7, 4))
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
bars = plt.bar(feature_names, importances, color=colors, edgecolor='black')
plt.title('Feature Importance', fontsize=13, fontweight='bold')
plt.ylabel('Importance Score')
plt.xlabel('Feature')
for bar, val in zip(bars, importances):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Feature importance saved as 'feature_importance.png'")


# ============================================================
# TASK 6: PREDICT AND PRESENT RESULTS
# ============================================================

print("\n[5] SAMPLE PREDICTIONS...\n")
print("-" * 55)

# Label encoders per column for prediction
encoders = {}
for col in cat_cols:
    enc = LabelEncoder()
    enc.fit(data[col])
    encoders[col] = enc

def predict_traffic(area, day, time_of_day, weather, is_holiday):
    input_data = pd.DataFrame([{
        'area'       : encoders['area'].transform([area])[0],
        'day'        : encoders['day'].transform([day])[0],
        'time_of_day': encoders['time_of_day'].transform([time_of_day])[0],
        'weather'    : encoders['weather'].transform([weather])[0],
        'is_holiday' : encoders['is_holiday'].transform([is_holiday])[0],
    }])
    prediction = model.predict(input_data)[0]
    label = encoders['congestion_level'].inverse_transform([prediction])[0]
    return label

# --- Test Cases ---
test_cases = [
    ('Saddar',    'Monday',    'Morning',   'Rainy',  'No'),
    ('DHA',       'Sunday',    'Night',     'Clear',  'Yes'),
    ('Clifton',   'Friday',    'Evening',   'Foggy',  'No'),
    ('Malir',     'Wednesday', 'Afternoon', 'Clear',  'No'),
    ('Gulshan',   'Tuesday',   'Morning',   'Hot',    'Yes'),
]

for i, (area, day, time, weather, holiday) in enumerate(test_cases, 1):
    result = predict_traffic(area, day, time, weather, holiday)
    print(f"Test {i}: {area}, {day}, {time}, {weather}, Holiday={holiday}")
    print(f"         --> Predicted Congestion: {result}\n")


# ============================================================
# FINAL CONCLUSION
# ============================================================

print("=" * 55)
print("   FINAL CONCLUSION")
print("=" * 55)
print(f"""
1. Dataset    : 500 synthetic records of Karachi traffic
               covering 8 areas, all days, 4 time slots,
               4 weather types, and holiday status.

2. Preprocessing: Missing values filled, duplicates removed,
               all categorical data label-encoded.
               Final dataset size: {len(data)} rows.

3. Visualization: Charts revealed that:
               - Evening & Morning have highest congestion
               - Rainy/Foggy weather worsens traffic
               - Saddar & Clifton are most congested areas

4. Model      : Random Forest Classifier (100 trees)
               Training: {len(X_train)} samples
               Testing : {len(X_test)} samples

5. Accuracy   : {accuracy * 100:.2f}% on test data

6. Key Finding: Time of Day and Weather are the most
               important features for predicting congestion.

7. Conclusion : This model can help citizens plan their
               commutes and avoid traffic jams in Karachi.
""")
print("=" * 55)
print("   PROJECT COMPLETE")
print("=" * 55)
