"""
Description:
This Python script applies stepwise logistic regression to classify hospital-acquired infections (HAI) with or without urinary tract infections (UTI). 
The script preprocesses the data using imputation, one-hot encoding, and scaling. It performs stepwise feature selection to identify significant predictors, 
calculates odds ratios using statsmodels, and evaluates model performance with metrics such as accuracy, specificity, sensitivity, and AUC. 
Additionally, it saves the results (coefficients, odds ratios) to an Excel file and plots the ROC curve to visualize performance.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load the data
file_path = r'C:\Users\...
data = pd.read_excel(file_path)

# Define predictors and target variable
X = data[['PATIENT_AGE', 'PATIENT_GENDER', 'PATIENT_ETHNICITY', 'PATIENT_EMPLOYMENT_STATUS', 
          'PATIENT_MARITAL_STATUS', 'Poverty', 'Median_Income', 
          'Graduate_degree', 'population_density', 'New Location', 'New Nurse Unit']]
y = data['H']  # Assuming 'H' is the target variable (binary: 0 = Non-UTI, 1 = UTI)

# Separate numeric and categorical features
numeric_features = ['PATIENT_AGE', 'Poverty', 'Median_Income', 'Graduate_degree', 'population_density']
categorical_features = ['PATIENT_GENDER', 'PATIENT_ETHNICITY', 'PATIENT_EMPLOYMENT_STATUS', 
                        'PATIENT_MARITAL_STATUS', 'New Location', 'New Nurse Unit']

# Impute missing values in numeric features
numeric_imputer = SimpleImputer(strategy='mean')
X_numeric_imputed = numeric_imputer.fit_transform(X[numeric_features])

# Impute missing values in categorical features
categorical_imputer = SimpleImputer(strategy='most_frequent')
X_categorical_imputed = categorical_imputer.fit_transform(X[categorical_features])

# Combine the imputed data back into a DataFrame
X_imputed = np.hstack((X_numeric_imputed, X_categorical_imputed))
X_imputed = pd.DataFrame(X_imputed, columns=numeric_features + categorical_features)

# Apply one-hot encoding to categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)])

# Apply the preprocessing
X_preprocessed = preprocessor.fit_transform(X_imputed)

# Convert to DataFrame for easier manipulation with statsmodels
X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=preprocessor.get_feature_names_out())

# Add a constant to the model for the intercept
X_preprocessed_df = sm.add_constant(X_preprocessed_df)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed_df, y, test_size=0.2, random_state=42)

# Define the stepwise logistic regression function
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out=0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.Logit
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit(disp=0)
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(disp=0)
        # use all coefficients except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() 
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

# Perform stepwise selection
resulting_features = stepwise_selection(X_train, y_train)

# Fit the final logistic regression model using statsmodels to get coefficients and odds ratios
final_model_sm = sm.Logit(y_train, sm.add_constant(X_train[resulting_features])).fit()

# Calculate the odds ratios and 95% confidence intervals
params = final_model_sm.params
conf = final_model_sm.conf_int()
conf['OR'] = np.exp(params)
conf.columns = ['2.5%', '97.5%', 'OR']

# Convert to DataFrame
summary_df = pd.DataFrame({
    'Coefficient': params,
    'OR': conf['OR'],
    '2.5%': conf['2.5%'],
    '97.5%': conf['97.5%']
})

# Round the numbers in the DataFrame to two decimal places
summary_df_rounded = summary_df.round(2)

# Save the rounded DataFrame to an Excel file
summary_df_rounded.to_excel('LR_HAI_2020.xlsx', index=True)

print("Summary and odds ratios (rounded to two decimal places) saved to the current directory as 'LR_HAI_2020.xlsx'.")

# Using sklearn's LogisticRegression with class weights
clf = LogisticRegression(class_weight={0: 1, 1: 10}, solver='liblinear')
clf.fit(X_train[resulting_features], y_train)

# Predict probabilities for the test set
y_pred_prob = clf.predict_proba(X_test[resulting_features])[:, 1]

# Predict binary outcomes based on the default threshold of 0.5
y_pred = clf.predict(X_test[resulting_features])

# Calculate and display classification metrics
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC: {roc_auc:.2f}")

# Extract confusion matrix values
tn, fp, fn, tp = conf_matrix.ravel()

# Calculate Specificity, Sensitivity, and Overall Accuracy
specificity = (tn / (tn + fp)) * 100
sensitivity = (tp / (tp + fn)) * 100
overall_accuracy = ((tp + tn) / (tp + tn + fp + fn)) * 100

print(f"Specificity: {specificity:.1f}")
print(f"Sensitivity: {sensitivity:.1f}")
print(f"Overall Accuracy: {overall_accuracy:.1f}")

# Generate a classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate and plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
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

