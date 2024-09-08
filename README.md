# UTI_MachineLearning_HAI_vs_CAI
**Comparative Analysis of Machine Learning Models for Predicting Hospital- and Community-Acquired Urinary Tract Infections
**Description
This project compares machine learning models to predict and distinguish between hospital-acquired (HAI) and community-acquired (CAI) urinary tract infections (UTIs). The study also differentiates between HAI patients with and without UTIs. The models use demographic, hospital, and socioeconomic predictors to enhance the accuracy of predictions and help healthcare providers implement timely interventions.

The following machine learning models were applied:

Decision Tree (DT)
Neural Network (NN)
Logistic Regression (LR)
Random Forest (RF)
Extreme Gradient Boosting (XGBoost)
This repository includes the code, data preprocessing steps, and model evaluations used in the analysis.

Project Goals
Compare the effectiveness of machine learning models in predicting HAI vs. CAI UTIs.
Identify key predictors such as nurse units, hospital locations, and socioeconomic factors.
Provide insights into the performance of different models, focusing on sensitivity, accuracy, and interpretability.
Installation Instructions
To run this project, you'll need the following installed:

Python 3.x
scikit-learn
pandas
numpy
matplotlib
xgboost
Usage
Data Preprocessing: The dataset from 2019-2023 was sourced from Cerner Corporation, with patient demographics, hospital locations, and socioeconomic factors. Preprocessing steps include handling missing values, recoding nurse units, and applying SMOTE for class imbalance.

Model Training: Run the scripts provided to train the models:

Logistic Regression: LR_HAI.py
Decision Tree: dDRT_HAI.py
Neural Network: NN_HAI.py
Random Forest: RF_HAI.py
XGBoost: xgboost_HAI.py
Evaluation: The models are evaluated using accuracy, sensitivity, specificity, F1 score, and AUC. Results are presented in tables and figures summarizing the performance across five years.

Results
Logistic Regression: Achieved the highest overall accuracy and AUC across most years.
Decision Tree: Showed the highest sensitivity, particularly useful for identifying positive cases, though it had lower precision.
Random Forest: Balanced performance with high precision and F1 score, excelling in cross-validation stability.
Neural Network: Demonstrated high specificity, effectively identifying negative cases.
XGBoost: Provided strong performance in AUC, especially with structured data, but required more computational resources.
