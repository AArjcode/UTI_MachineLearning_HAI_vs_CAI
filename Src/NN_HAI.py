"""
Description:
This Python script builds and trains a Neural Network to classify hospital-acquired infections (HAI) with or without urinary tract infections (UTI) using demographic, hospital, and socioeconomic data. It includes data preprocessing (scaling, one-hot encoding, imputation), addresses class imbalance using SMOTE and class weights, and evaluates the model using accuracy, AUC, specificity, and confusion matrix. The script also visualizes performance using a ROC curve.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Load the data
file_path = r'C:\Users\...
data = pd.read_excel(file_path)

# Define predictors and target variable
X = data[['PATIENT_AGE', 'PATIENT_GENDER', 'PATIENT_ETHNICITY', 'PATIENT_EMPLOYMENT_STATUS', 
          'PATIENT_MARITAL_STATUS', 'Poverty', 'Median_Income', 
          'Graduate_degree', 'population_density', 'New Location', 'New Nurse Unit']]
y = data['H']

# Preprocessing for categorical data
categorical_features = ['PATIENT_GENDER', 'PATIENT_ETHNICITY', 'PATIENT_EMPLOYMENT_STATUS', 
                        'PATIENT_MARITAL_STATUS', 'New Location', 'New Nurse Unit']
numeric_features = ['PATIENT_AGE', 'Poverty', 'Median_Income', 'Graduate_degree', 'population_density']

# OneHotEncoder for categorical variables and StandardScaler for numeric variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply the preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Impute missing values in numeric features
imputer = SimpleImputer(strategy='mean')  # You can change the strategy to 'median' or 'most_frequent' as needed
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)

# Calculate class weights to handle imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)
class_weights = {i : class_weights[i] for i in range(len(class_weights))}

# Define the model
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train_resampled.shape[1]))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model without cross-validation
history = model.fit(X_train_resampled, y_train_resampled, epochs=50, batch_size=32, 
                    validation_data=(X_test_imputed, y_test), class_weight=class_weights)

# Predict the test set results with an adjusted threshold
y_pred = (model.predict(X_test_imputed) > 0.1).astype("int32")  # Adjust the threshold 

# Calculate AUC
roc_auc = roc_auc_score(y_test, y_pred)
print(f'AUC: {roc_auc}')

# Generate a classification report
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print(classification_report(y_test, y_pred))

# Calculate Specificity and Overall Accuracy
tn, fp, fn, tp = conf_matrix.ravel()
specificity = (tn / (tn + fp)) * 100
overall_accuracy = ((tp + tn) / (tp + tn + fp + fn)) * 100

print(f"Specificity: {specificity:.1f}")
print(f"Overall Accuracy: {overall_accuracy:.1f}")

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict(X_test_imputed))

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

