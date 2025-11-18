# Heart Disease Prediction: Handling Imbalanced Data 

## Project Overview
This project focuses on a critical medical diagnostic task: predicting whether a patient has heart disease. The core challenge of this dataset is **Class Imbalance**—the number of healthy patients far exceeds those with heart disease. A standard model might achieve high accuracy by simply guessing "Healthy" for everyone, but this would fail the primary medical objective: identifying sick patients.

Therefore, this project prioritizes **Recall (Sensitivity)** over Accuracy to minimize false negatives (missed diagnoses).

## Key Objectives
- **Advanced Preprocessing**: Implementing a complex encoding strategy for mixed data types (Binary, Ordinal, Nominal).
- **Handling Imbalance**: Using **Stratified Splitting** and **Class Weights** to force models to pay attention to the minority class.
- **Leakage Prevention**: Strictly separating training and testing data during Scaling and Transformation.
- **Hyperparameter Tuning**: Using **GridSearchCV** to optimize models specifically for the Recall metric.

## Tech Stack & Methods
- **Python**: Pandas, NumPy, Scikit-Learn, XGBoost,RandomForest
- **Feature Engineering**:
    - `OneHotEncoder` (drop='if_binary') for binary features.
    - `OrdinalEncoder` with manual mapping for ordered features (e.g., Health ratings).
    - `OneHotEncoder` (drop='first') for nominal features to prevent multicollinearity.
- **Modeling**: Logistic Regression, Random Forest, XGBoost.
- **Optimization**: `GridSearchCV` with `scoring='recall'`.

## Workflow & Results

1.  **Data Encoding Strategy**:
    - Separated features into Binary, Ordinal, and Nominal types and applied specific encoding pipelines for each to preserve maximum information.

2.  **Combatting Imbalance**:
    - Calculated the `class_ratio` (Negative/Positive samples).
    - Applied `class_weight='balanced'` in Logistic Regression and Random Forest.
    - Applied `scale_pos_weight` in XGBoost to heavily penalize misclassifying positive cases.

3.  **Grid Search & Tuning**:
    - Tuned a Random Forest model optimizing for **Recall**.
    - **Result**: The tuned model achieved a **Recall of ~79.25%** on the unseen test set, with consistent performance across cross-validation folds (78.90%), indicating a robust model with no overfitting.
