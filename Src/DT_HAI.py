
"""
Description:
This Python script applies a Decision Tree Classifier to predict and distinguish hospital-acquired (HAI) vs. community-acquired (CAI) urinary tract infections using demographic, hospital, and socioeconomic data. 
The script includes preprocessing steps such as handling missing data, one-hot encoding, scaling, and applying SMOTE for class imbalance. 
It uses grid search with cross-validation, evaluates performance using metrics like accuracy, specificity, ROC-AUC, and visualizes the decision tree. 
Feature importance is calculated and saved to an Excel file.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
# Import necessary modules for plotting the decision tree
from sklearn.tree import plot_tree

# Load the data
file_path = r'C:\Users....
data = pd.read_excel(file_path)

# Define predictors and target variable
X = data[['PATIENT_AGE', 'PATIENT_GENDER', 'PATIENT_ETHNICITY', 'PATIENT_EMPLOYMENT_STATUS', 
          'PATIENT_MARITAL_STATUS', 'Poverty', 'Median_Income', 
          'Graduate_degree', 'population_density', 'New Location', 'New Nurse Unit']]
y = data['H']  # Assuming 'H' is the target variable (binary: 0 = Non-UTI, 1 = With UTI)

# Handle missing values with imputation
numeric_features = ['PATIENT_AGE', 'Poverty', 'Median_Income', 'Graduate_degree', 'population_density']
categorical_features = ['PATIENT_GENDER', 'PATIENT_ETHNICITY', 'PATIENT_EMPLOYMENT_STATUS', 
                        'PATIENT_MARITAL_STATUS', 'New Location', 'New Nurse Unit']

numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])
X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])

# Apply one-hot encoding to categorical data
one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
X_categorical_encoded = one_hot_encoder.fit_transform(X[categorical_features])

# Combine encoded categorical and numeric data
X_combined = np.hstack((X[numeric_features].values, X_categorical_encoded))

# Convert combined array back to DataFrame with appropriate column names
encoded_cat_columns = one_hot_encoder.get_feature_names_out(categorical_features)
X_combined_df = pd.DataFrame(X_combined, columns=numeric_features + list(encoded_cat_columns))

# Apply standard scaling for numeric data
scaler = StandardScaler()
X_combined_df[numeric_features] = scaler.fit_transform(X_combined_df[numeric_features])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined_df, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize the Decision Tree Classifier with class weights
model = DecisionTreeClassifier(random_state=42, class_weight={0: 1, 1: 10})  # Adjust the weight for the minority class

# Define parameters for grid search
params = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 5}, {0: 1, 1: 10}]
}

# Perform grid search with cross-validation using all CPU cores
grid_search = GridSearchCV(model, params, scoring='recall', cv=5, n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best model from grid search
best_crt_model = grid_search.best_estimator_

# Cross-validation on the final model
cross_val_scores = cross_val_score(best_crt_model, X_train_resampled, y_train_resampled, cv=5)
print("\nCross-Validation Accuracy Scores:", cross_val_scores)
print("Mean Cross-Validation Accuracy:", np.mean(cross_val_scores))

# Make predictions on the test set
y_pred = best_crt_model.predict(X_test)
y_prob = best_crt_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Best Parameters:", grid_search.best_params_)
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate the ROC-AUC score
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc}")

# Calculate Specificity and Overall Accuracy
tn, fp, fn, tp = conf_matrix.ravel()
specificity = (tn / (tn + fp)) * 100
overall_accuracy = ((tp + tn) / (tp + tn + fp + fn)) * 100

print(f"Specificity: {specificity:.1f}")
print(f"Overall Accuracy: {overall_accuracy:.1f}")

# Calculate and save the normalized importance factors to an Excel file
importances = best_crt_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_combined_df.columns,
    'Importance (%)': np.round(importances * 100, 2)
})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance (%)', ascending=False)

# Save the DataFrame to an Excel file in the current directory
importance_df.to_excel('Feature_Importance_HAI_2019.xlsx', index=False)

print("Normalized importance factors saved to 'Feature_Importance_HAI_2019.xlsx' in the current directory.")
# Limit the depth of the best model for visualization purposes
best_crt_model_for_plotting = DecisionTreeClassifier(max_depth=4, random_state=42)
best_crt_model_for_plotting.fit(X_train_resampled, y_train_resampled)

# Plot the tree
plt.figure(figsize=(20, 10))
plot_tree(best_crt_model_for_plotting, 
          feature_names=X_combined_df.columns, 
          class_names=['CAI', 'HAI'], 
          filled=True, 
          rounded=True, 
          precision=2, 
          fontsize=12, 
          proportion=True,  # Display probabilities instead of counts
          label='root')  # Only display the condition and class
plt.title("Decision Tree Visualization (Max Depth = 4)")
plt.show()

print(y.value_counts())
print(best_crt_model.classes_)

