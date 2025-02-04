# Import necessary libraries
import numpy as np
import pandas as pd
import pickle  # For saving the trained model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score

# Load dataset (Update the path if needed)
DATA_PATH = 'vocal_gender_features_new.csv'

try:
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully. Shape:", df.shape)
except FileNotFoundError:
    print(f"Error: File not found at {DATA_PATH}. Please check the file path.")
    exit()

# Create target variable (True for Male, False for Female)
df['Male'] = df['label'] == 1  

# Define feature columns (same as used in Flask app)
FEATURE_COLUMNS = [
    'mean_spectral_centroid', 'std_spectral_centroid', 'zero_crossing_rate',
    'rms_energy', 'mean_pitch'
]

# Extract features and target
X = df[FEATURE_COLUMNS]
y = df['Male']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2024, stratify=y
)

# Train logistic regression model
logreg = LogisticRegression(max_iter=10000, tol=1e-12, random_state=2024)
logreg.fit(X_train, y_train)

# Evaluate model
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and scaler for Flask API
with open("logistic_regression_model.pkl", "wb") as model_file:
    pickle.dump(logreg, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("\nModel and scaler saved successfully!")
DATA_PATH = 'vocal_gender_features_new.csv'

try:
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully. Shape:", df.shape)
except FileNotFoundError:
    print(f"Error: File not found at {DATA_PATH}. Please check the file path.")
    exit()

# Create target variable (True for Male, False for Female)
df['Male'] = df['label'] == 1  

# Define feature columns (same as used in Flask app)
FEATURE_COLUMNS = [
    'mean_spectral_centroid', 'std_spectral_centroid', 'zero_crossing_rate',
    'rms_energy', 'mean_pitch'
]

# Extract features and target
X = df[FEATURE_COLUMNS]
y = df['Male']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2024, stratify=y
)

# Train logistic regression model
logreg = LogisticRegression(max_iter=10000, tol=1e-12, random_state=2024)
logreg.fit(X_train, y_train)

# Evaluate model
y_pred = logreg.predict(X_test)
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and scaler for Flask API
with open("logistic_regression_model.pkl", "wb") as model_file:
    pickle.dump(logreg, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("\nModel and scaler saved successfully!")
