# Personalized Music Recommendation Influence Analysis

## Project Overview
This project explores the development of a personalized music recommendation system using data from the **KKBOX Music Recommendation Challenge**. The primary objective is to predict whether a user will replay a specific song within one month of their first listen, effectively modeling this as a binary classification problem.

Beyond standard predictive modeling, this project implements **Influence Estimation** techniques (Leave-One-Out and Group-Level Influence) to analyze model robustness and determine how specific training data points or subgroups affect model decision-making.

## Key Results & Insights
* **Best Model**: The **Random Forest Classifier** achieved the highest performance with a training accuracy of **86.4%** and a testing accuracy of **62.5%**.
* **Model Comparison**:
    * **LightGBM**: Demonstrated strong generalization with **66.5%** testing accuracy.
    * **Naive Bayes & Logistic Regression**: Struggled to capture complex non-linear relationships, yielding results close to random guessing (~50% accuracy).
* **Influence Analysis**:
    * **Random Forest** showed high sensitivity to specific training points (high LOO influence scores), suggesting it relies heavily on specific "power user" subgroups.
    * **Naive Bayes** proved to be the most robust to data removal but lacked predictive power due to its independence assumptions.

## Methodology
The project follows a standard 5-step data processing pipeline:

### 1. Data Preprocessing & ETL
* **Data Cleaning**: Handled missing values through median/mode imputation and removal of sparse rows. Outliers in user age and song length were normalized.
* **Feature Engineering**: Extracted primary genres from complex strings, calculated membership duration, and standardized artist names.
* **Memory Optimization**: Converted categorical variables (e.g., IDs) to optimized data types to handle large datasets efficiently.

### 2. Machine Learning Modeling
Four models were trained and evaluated on an 80/20 train-test split:
* **Gaussian Naive Bayes**: Baseline model.
* **Logistic Regression**: Selected for interpretability and probability outputs.
* **Random Forest**: Chosen for its ability to handle mixed data types and non-linear relationships.
* **LightGBM**: Implemented for high efficiency and gradient boosting performance.

### 3. Influence Estimation
* **Leave-One-Out (LOO)**: Systematically removed individual training examples to measure their impact on the AUROC score, identifying influential outliers.
* **Group Influence**: Ablated random subsets of data (5% to 55%) to test model stability against data loss.

## Repository Structure

| File | Description |
| :--- | :--- |
| `kkbox_data_preprocessing.py` | Handles initial ETL, memory optimization, outlier removal, and data cleaning. |
| `data_merging.py` | Merges user, song, and interaction datasets; handles missing value imputation for the combined set. |
| `data_modelling.py` | Trains and evaluates ML models (NB, LR, RF, LightGBM). Generates confusion matrices and ROC scores. |
| `LOO_influence.py` | Performs Leave-One-Out influence analysis to identify high-leverage data points. |
| `group_influence.py` | Conducts group-level influence analysis to assess model stability under data reduction. |

## Technologies Used
* **Python**: Core programming language.
* **Data Manipulation**: Pandas, NumPy.
* **Machine Learning**: Scikit-Learn, LightGBM.
* **Visualization**: Matplotlib, Seaborn.

## Data Source
The dataset is sourced from the WSDM 2018 KKBOX Music Recommendation Challenge on Kaggle:
[KKBOX Music Recommendation Challenge](https://www.kaggle.com/c/kkbox-music-recommendation-challenge)

## Contributors
* **Niroshan Benjamin**: Data sourcing, data modeling, performance evaluation, and influence analysis.
* **Jennifer Hung**: Data preprocessing, cleaning, and implementation of influence score algorithms.
* **Abdullah Naeem**: Data cleaning, modeling, and report documentation.
