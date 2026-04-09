# Pima Indians Diabetes Prediction Project

## Overview

This project implements multiple machine learning models to predict the onset of diabetes in Pima Indian women based on diagnostic measurements. The dataset originates from the National Institute of Diabetes and Digestive and Kidney Diseases and contains medical predictor variables and one target variable (Outcome).

## Dataset Features

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (mg/dL)
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)²)
- **DiabetesPedigreeFunction**: Diabetes pedigree function (likelihood of diabetes based on family history)
- **Age**: Age in years
- **Outcome**: Class variable (0 = non-diabetic, 1 = diabetic)

## Models Implemented

1. **Logistic Regression** - Linear model for binary classification
2. **Random Forest** - Ensemble of 300 decision trees
3. **Support Vector Machine (SVM)** - RBF kernel for non-linear separation
4. **K-Nearest Neighbors (KNN)** - Classification based on 5 nearest neighbors
5. **XGBoost** - Gradient boosting with optimized hyperparameters

## Data Preprocessing

- **Handling invalid zeros**: Features like Glucose, BloodPressure, SkinThickness, Insulin, and BMI cannot have zero values medically. These were replaced with column medians.
- **Train-test split**: 70-30 split with stratification to maintain class proportions
- **Feature scaling**: StandardScaler applied to normalize features (mean=0, std=1) for distance-based models

## Key Findings

- **Random Forest** achieved the highest ROC AUC score (0.844)
- **XGBoost** achieved the highest accuracy (77.1%)
- All models showed better precision for non-diabetic cases (class 0) than diabetic cases (class 1)
- Glucose was identified as the most important feature in XGBoost, which aligns with real-world medical knowledge

## Model Performance

| Model               | Accuracy | ROC AUC | Precision (0/1) | Recall (0/1) |
| ------------------- | -------- | ------- | --------------- | ------------ |
| Logistic Regression | 73.2%    | 0.841   | 0.90 / 0.58     | 0.66 / 0.86  |
| Random Forest       | 75.3%    | 0.844   | 0.88 / 0.61     | 0.72 / 0.81  |
| SVM (RBF)           | 73.6%    | 0.824   | 0.87 / 0.59     | 0.70 / 0.80  |
| KNN                 | 71.0%    | 0.777   | 0.87 / 0.56     | 0.65 / 0.81  |
| XGBoost             | 77.1%    | 0.837   | 0.89 / 0.63     | 0.73 / 0.84  |

## Technologies Used

- Python 3.12
- pandas, numpy for data manipulation
- scikit-learn for machine learning models and preprocessing
- matplotlib, seaborn for visualization
- xgboost for gradient boosting

## Sample Predictions

The model can predict diabetes risk for new patient data. High glucose, age, and BMI equates to a high chance of diabetes. Normal values and borderline profiles are considered normal (non-diabetic)

## Project Structure

- Data loading and exploration
- Handling missing/invalid values
- Feature/target separation and train-test split
- Feature scaling (for the benefit of models that use it such as logistic regression)
- Model training (5 different classifiers)
- Model evaluation with accuracy, ROC AUC, and classification reports
- ROC curve visualization
- Feature importance analysis for XGBoost which is the best model so far

## Usage

Run the Jupyter notebook sequentially to:

1. Load and preprocess the data
2. Train all five models
3. Compare performance metrics
4. Visualize results
5. Make predictions on new patient data

## Future Improvements

this is just the beginning. I am not yet satisfied with the recall values and I'll be actively working to improve them. I plan to incorporate the following:

- Hyperparameter tuning using GridSearchCV ---- (ongoing)
- Feature engineering (binning age, creating interaction terms, maybe more)
- Cross-validation for more robust evaluation
- Ensemble methods combining multiple models
- Increasing training data size (currently less than 768. Not optimal for a medical model.)
