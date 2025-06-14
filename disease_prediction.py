# -*- coding: utf-8 -*-
"""
Disease Prediction with Random Forest (Clean Version with Manual Upload)
"""

# ---- 1. Install Required Libraries (for Colab) ----
!pip install -q pandas scikit-learn imbalanced-learn joblib

# ---- 2. Import Dependencies ----
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
from google.colab import files

# ---- 3. Upload CSV File ----
print("Please upload your dataset file (CSV format)...")
uploaded = files.upload()

# ---- 4. Load Dataset ----
for filename in uploaded.keys():
    df = pd.read_csv(filename)

# ---- 5. Handle Missing Values ----
df.fillna(df.mode().iloc[0], inplace=True)

# ---- 6. Encode Categorical Columns ----
label_encoders = {}
categorical_cols = ['Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing',
                    'Gender', 'Blood Pressure', 'Cholesterol Level', 'Outcome Variable']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ---- 7. Feature Scaling ----
scaler = StandardScaler()
df['Age'] = scaler.fit_transform(df[['Age']])

# ---- 8. Feature and Target Split ----
X = df.drop('Outcome Variable', axis=1)
y = df['Outcome Variable']

# ---- 9. Handle Class Imbalance ----
X, y = SMOTE(random_state=42, k_neighbors=1).fit_resample(X, y)

# ---- 10. Train-Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---- 11. Train Random Forest Model ----
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ---- 12. Save the Model and Transformers ----
joblib.dump(model, 'rf_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')

# ---- 13. Evaluate Model with Human-Readable Labels ----
y_pred = model.predict(X_test)
outcome_encoder = label_encoders['Outcome Variable']
y_test_decoded = outcome_encoder.inverse_transform(y_test)
y_pred_decoded = outcome_encoder.inverse_transform(y_pred)

print(f"\n Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\n📊 Classification Report (with actual labels):")
print(classification_report(y_test_decoded, y_pred_decoded))
