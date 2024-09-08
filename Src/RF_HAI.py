"""
Description:
This Python script builds and trains a Random Forest Classifier to classify hospital-acquired infections (HAI) with or without urinary tract infections (UTI). 
The script preprocesses the data using one-hot encoding, scaling, and SMOTE for handling class imbalance. 
It evaluates model performance at various thresholds using accuracy, specificity, precision, sensitivity, F1 score, and AUC. 
Additionally, it performs cross-validation and plots the ROC curve to visualize performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the data (update file_path as per your directory structure)
file_path = r'C:\Users\...
data = pd.read_excel(file_path)

# Define predictors and target variable
X = data[['PATIENT_AGE', 'PATIENT_GENDER', 'PATIENT_ETHNICITY', 'PATIENT_EMPLOYMENT_STATUS', 
          'PATIENT_MARITAL_STATUS', 'Poverty', 'Median_Income', 
          'Graduate_degree', 'population_density', 'New Location', 'New Nurse Unit']]
y = data['H']  # Assuming 'H' is the target variable (binary: 0 = Non-UTI, 1 = UTI)

# Handle missing values with imputation
from sklearn.impute import SimpleImputer
numeric_features = ['PATIENT_AGE', 'Poverty', 'Median_Income', 'Graduate_degree', 'population_density']
categorical_features = ['PATIENT_GENDER', 'PATIENT_ETHNICITY', 'PATIENT_EMPLOYMENT_STATUS', 
                        'PATIENT_MARITAL_STATUS', 'New Location', 'New Nurse Unit']

numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])
X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])

# Apply one-hot encoding to categorical data
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
X_categorical_encoded = one_hot_encoder.fit_transform(X[categorical_features])

# Combine encoded categorical and numeric data
X_combined = np.hstack((X[numeric_features].values, X_categorical_encoded))

# Convert combined array back to DataFrame with appropriate column names
encoded_cat_columns = one_hot_encoder.get_feature_names_out(categorical_features)
X_combined_df = pd.DataFrame(X_combined, columns=numeric_features + list(encoded_cat_columns))

# Apply standard scaling for numeric data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_combined_df[numeric_features] = scaler.fit_transform(X_combined_df[numeric_features])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined_df, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize the RandomForest with class weights and enable all CPU cores
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0: 1, 1: 10}, n_jobs=-1)

# Train the model
rf_model.fit(X_train_resampled, y_train_resampled)

# Get predicted probabilities
y_prob = rf_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Define a function to calculate metrics, AUC, and cross-validation accuracy for different thresholds
def evaluate_threshold_with_cv(threshold):
    # Apply the threshold
    y_pred_threshold = (y_prob >= threshold).astype(int)
    
    # Calculate confusion matrix and extract metrics
    conf_matrix = confusion_matrix(y_test, y_pred_threshold)
    TN, FP, FN, TP = conf_matrix.ravel()
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
    specificity = TN / (TN + FP) * 100
    precision = TP / (TP + FP) * 100
    sensitivity = TP / (TP + FN) * 100
    f1 = f1_score(y_test, y_pred_threshold) * 100
    auc = roc_auc_score(y_test, y_prob) * 100
    
    # Perform cross-validation on the Random Forest model
    cross_val_scores = cross_val_score(rf_model, X_train_resampled, y_train_resampled, cv=5, n_jobs=-1)
    cross_val_accuracy = np.mean(cross_val_scores) * 100

    # Print results for the current threshold
    print(f"\nThreshold: {threshold:.2f}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Specificity: {specificity:.1f}%")
    print(f"Precision: {precision:.1f}%")
    print(f"Sensitivity: {sensitivity:.1f}%")
    print(f"F1 Score: {f1:.1f}%")
    print(f"AUC: {auc:.1f}%")
    print(f"Cross-Validation Accuracy: {cross_val_accuracy:.1f}%")

# Experiment with different thresholds and calculate cross-validation accuracy
for threshold in [0.5, 0.4, 0.3, 0.2, 0.1]:
    evaluate_threshold_with_cv(threshold)

# Optionally, plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


