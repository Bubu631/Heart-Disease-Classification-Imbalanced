# Heart Disease Prediction 

## Project Overview
This project focuses on a critical medical diagnostic task: predicting heart disease (a binary classification problem). The core challenge is the severe **Class Imbalance** in the dataset, where healthy samples significantly outnumber positive cases. A standard model would achieve misleadingly high accuracy by simply ignoring the minority class.

Therefore, this project strictly prioritizes **Recall (Sensitivity)** over Accuracy to minimize **False Negatives** (missed diagnoses), which is the most critical factor in medical screening.

## Key Objectives
The project structure was designed to systematically address the challenges of mixed data types and class imbalance:

* **Advanced Preprocessing:** Implementing a specialized encoding strategy to preserve feature integrity.
* **Ordinality:** Using `OrdinalEncoder` with manual ranking (e.g., Health ratings) to preserve the **intrinsic logical order** of features.
* **Nominality:** Using `OneHotEncoder(drop='first')` to prevent the **Dummy Variable Trap (multicollinearity)** for unordered features (like Race).
* **Leakage Prevention:** Strictly separating training and testing data during Scaling and Transformation parameters (`StandardScaler`) to prevent **data leakage**.
* **Handling Imbalance:** Utilizing **Stratified Splitting** and **Class Weights** (`'balanced'` or `scale_pos_weight`) to force models to learn from the sparse minority class.
* **Hyperparameter Tuning:** Employing **GridSearchCV** to optimize models explicitly for the **Recall metric**, ensuring the best possible sensitivity.

##  Tech Stack & Methods

* **Python Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, RandomForest
* **Data Preparation:** `OneHotEncoder`, `OrdinalEncoder`, `StandardScaler`, `StratifiedKFold`
* **Modeling Strategy:** Comparing Linear (Logistic Regression) and Non-Linear (XGBoost, Random Forest) classifiers.
* **Optimization:** `GridSearchCV` with `scoring='recall'`.

##  Workflow & Comparative Results

### Encoding and Imbalance Strategy

* The Data Encoding Strategy separated features into Binary, Ordinal, and Nominal types, applying the most appropriate encoding for each.
* The Imbalance Strategy was implemented by applying `class_weight='balanced'` in Logistic Regression/Random Forest and `scale_pos_weight=class_ratio` in XGBoost, effectively increasing the penalty for missing a positive case.

### Final Model Comparison Summary

The results confirmed that complex, non-linear models significantly outperformed the linear baseline and proved the effectiveness of the class-weighting strategy.

| Model | Testing Recall (Sensitivity) | Testing Precision | Analysis |
| :--- | :--- | :--- | :--- |
| **XGBoost Classifier** | **79.72% (Highest)** | 21.86% | **Winner.** The boosting mechanism provided the highest capability in identifying minority samples. |
| **Random Forest (Tuned)** | 79.25% | 22.01% (Approx) | **Highly Competitive.** After Hyperparameter Tuning, the model's performance was elevated by ~2.4% over the manual setup, proving the value of optimization. |
| **Logistic Regression (Weighted)** | 77.77% (Baseline) | 22.46% | Good stability, but insufficient complexity to match tree-based models. |

---

## Final Conclusion and Recommendation

The recommended model for deployment is the **XGBoost Classifier**.

It successfully achieved the **highest Recall (79.72%)** on the unseen test data. This performance is crucial for a medical application where minimizing **False Negatives** is the paramount objective. The project validates that effective machine learning for real-world problems requires not just complex models, but meticulous preprocessing, strategic evaluation, and targeted optimization to overcome data deficiencies like class imbalance.
