import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import numpy as np
import gc  # Import garbage collection
import seaborn as sns
import matplotlib.pyplot as plt

# Define a smaller sample size
SAMPLE_SIZE = 1000  # Reduced sample size (change if killing terminal)

# Load data with optimized dtypes
merged = pd.read_csv('final_merged.csv',
                    dtype={
                        'song_length': 'int32',
                        'bd': 'int16',
                        'membership_days': 'int32',
                        'target': 'int8'
                    },
                    parse_dates=['registration_init_time', 'expiration_date'],
                    low_memory=True)

# Replace string "nan" with np.nan BEFORE ANY CONVERSIONS
merged = merged.replace("nan", np.nan)

# Determine object columns *before* any conversions
object_cols = [col for col in merged.columns if merged[col].dtype == 'object']

# Ensure 'registration_init_time' and 'expiration_date' are NOT in object_cols
object_cols = [col for col in object_cols if col not in ['registration_init_time', 'expiration_date']]

# Convert object columns to category *before* numeric conversion
for col in object_cols:
    try:
        merged[col] = merged[col].astype('category')
    except:
        pass

# Convert datetime columns to numeric (timestamp), handling NaT
merged['registration_init_time'] = (merged['registration_init_time'].fillna(0).astype(np.int64) // 10**9).astype('int64')
merged['expiration_date'] = (merged['expiration_date'].fillna(0).astype(np.int64) // 10**9).astype('int64')

# One-Hot Encode Categorical Features BEFORE sampling
categorical_cols = [col for col in merged.columns if merged[col].dtype == 'object' or merged[col].dtype == 'category']

# Sample BEFORE one hot encoding
sampled_merged = merged.sample(n=SAMPLE_SIZE, random_state=42)

# Now do one hot encoding
sampled_merged = pd.get_dummies(sampled_merged, columns=categorical_cols)

# Split features and target variable
X = sampled_merged.drop(columns=['target'])
y = sampled_merged['target']

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Gaussian Naive Bayes
Bayes_model = GaussianNB()
Bayes_model.fit(X_train, y_train)
Bayes_probs = Bayes_model.predict_proba(X_valid)[:, 1]
Bayes_roc = roc_auc_score(y_valid, Bayes_probs)
print("Naive Bayes Accuracy Score for Training data: ", Bayes_model.score(X_train, y_train))
print("Naive Bayes Accuracy Score for Test data: ", Bayes_model.score(X_valid, y_valid))
print('Bayes ROC Score: ', Bayes_roc)


# Logistic Regression 
Logistic_Regression_model = LogisticRegression(max_iter = 1200, random_state = 42)
Logistic_Regression_model.fit(X_train, y_train)
Logistic_Regression_probs = Logistic_Regression_model.predict_proba(X_valid)[:, 1]
Logistic_roc = roc_auc_score(y_valid, Logistic_Regression_probs)
print("Logistic Regression Accuracy Score for Training data: ", Logistic_Regression_model.score(X_train, y_train))
print("Logistic Regression Accuracy Score for Test data: ", Logistic_Regression_model.score(X_valid, y_valid))
print('Logistic Regression ROC Score: ', Logistic_roc)

# Random Forest Classifier
RF_model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10, random_state=42)
RF_model.fit(X_train, y_train)
RF_probs = RF_model.predict_proba(X_valid)[:, 1]
RF_roc = roc_auc_score(y_valid, RF_probs)
print("Random Forest Accuracy Score for Training data: ", RF_model.score(X_train, y_train))
print("Random Forest Accuracy Score for Test data: ", RF_model.score(X_valid, y_valid))
print('Random Forest ROC Score: ', RF_roc)

# Confusion Matrix for RF Classifier - best performing model
y_prediction = RF_model.predict(X_valid)
conf_matrix = confusion_matrix(y_valid, y_prediction)
sns.set(font_scale = 1.4)
plt.figure(figsize = (8,6))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'YlGnBu', xticklabels = RF_model.classes_, yticklabels = RF_model.classes_)

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')   
plt.title('Confusion Matrix for Random Forest Classifier')
plt.savefig('Confusionmatrix_RF')

# LightGBM Implementation added below
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
import re

import re

def rename_columns_unique(columns):
    seen = set()
    new_columns = []
    for col in columns:
        new_col = re.sub(r'[^\w]', '_', col)
        original_col = new_col
        i = 1
        while new_col in seen:
            new_col = f"{original_col}_{i}"
            i += 1
        seen.add(new_col)
        new_columns.append(new_col)
    return new_columns

# Apply unique renaming
X_train.columns = rename_columns_unique(X_train.columns)
X_valid.columns = rename_columns_unique(X_valid.columns)

# Align columns
X_valid = X_valid[X_train.columns]

# Create LightGBM datasets
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

# LightGBM parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': 0,
    'random_state': 42
}

gbm_model = lgb.train(
    params,
    lgb_train,
    num_boost_round=100,
    valid_sets=[lgb_eval],
    callbacks=[lgb.early_stopping(10)]
)

# Predict and Evaluate
lgb_probs = gbm_model.predict(X_valid, num_iteration=gbm_model.best_iteration)
lgb_preds = (lgb_probs > 0.5).astype(int)

train_preds = gbm_model.predict(X_train, num_iteration=gbm_model.best_iteration)
train_preds_binary = (train_preds > 0.5).astype(int)

print("LightGBM Accuracy Score for Training data: ", accuracy_score(y_train, train_preds_binary))
print("LightGBM Accuracy Score for Test data: ", accuracy_score(y_valid, lgb_preds))
print("LightGBM ROC Score: ", roc_auc_score(y_valid, lgb_probs))


# Clean up memory
del merged, sampled_merged, X, y, X_train, X_valid, y_train, y_valid
gc.collect()

