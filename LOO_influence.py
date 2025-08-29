from data_modelling import *
import random
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

#LOO Influence

#Calculating Baseline AUROC Scores

# Gaussian Naive Bayes
y_prob_nb = Bayes_model.predict_proba(X_valid)[:, 1]
baseline_auc_nb = roc_auc_score(y_valid, y_prob_nb)

# Logistic Regression 
y_prob_lr = Logistic_Regression_model.predict_proba(X_valid)[:, 1]
baseline_auc_lr = roc_auc_score(y_valid, y_prob_lr)

# Random Forest Classifier
y_prob_rf = RF_model.predict_proba(X_valid)[:, 1]
baseline_auc_rf = roc_auc_score(y_valid, y_prob_rf)

#LOO Influence Calculation
random.seed(56)
# randomly select training observations from "X_train" 
available_observations = range(len(X_train))
random_indices = random.sample(list(available_observations), 10)

# create an array to store influence scores 
influence_scores_nb = []
influence_scores_lr = []
influence_scores_rf = []

for i in random_indices:
    # Remove one point
    X_train_loo = np.delete(X_train, i, axis=0)
    y_train_loo = np.delete(y_train, i, axis=0)

    # Retrain Naïve Bayes
    model_nb_loo = GaussianNB()
    model_nb_loo.fit(X_train_loo, y_train_loo)
    y_prob_loo_nb = model_nb_loo.predict_proba(X_valid)[:, 1]
    new_auc_nb = roc_auc_score(y_valid, y_prob_loo_nb)
    
    # Retrain Logistic Regression
    model_lr_loo = LogisticRegression(max_iter = 1200, random_state = 42)
    model_lr_loo.fit(X_train_loo, y_train_loo)
    y_prob_loo_lr = model_lr_loo.predict_proba(X_valid)[:, 1]
    new_auc_lr = roc_auc_score(y_valid, y_prob_loo_lr)


    # Retrain Random Forest
    model_rf_loo = RandomForestClassifier(n_estimators=350, max_depth=5)
    model_rf_loo.fit(X_train_loo, y_train_loo)
    y_prob_loo_rf = model_rf_loo.predict_proba(X_valid)[:, 1]
    new_auc_rf = roc_auc_score(y_valid, y_prob_loo_rf)

    # Influence Calculation
    influence_nb = baseline_auc_nb - new_auc_nb
    influence_lr = baseline_auc_lr - new_auc_lr
    influence_rf = baseline_auc_rf - new_auc_rf


    influence_scores_nb.append((i, influence_nb))
    influence_scores_lr.append((i,influence_lr))
    influence_scores_rf.append((i, influence_rf))

    print(f"Point {i}: Influence on NB = {influence_nb:.6f}, Influence on LR = {influence_lr:.6f}, Influence on RF = {influence_rf:.6f}")

#Convert to DataFrame
influence_df = pd.DataFrame({
    'Point Index': [i[0] for i in influence_scores_nb],
    'Influence NB': [i[1] for i in influence_scores_nb],
    'Influence LR': [i[1] for i in influence_scores_lr],
    'Influence RF': [i[1] for i in influence_scores_rf]
})

print(influence_df)

plt.figure(figsize=(10, 4))
plt.scatter(influence_df['Point Index'], influence_df['Influence RF'], label='Random Forest', color='red')
plt.scatter(influence_df['Point Index'], influence_df['Influence LR'], label='Logistic Regression', color='green')
plt.scatter(influence_df['Point Index'], influence_df['Influence NB'], label='Naive Bayes', color='blue', alpha=0.7)  # Added NB
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Point Index')
plt.ylabel('Influence (ΔAUROC)')
plt.legend()
plt.title('LOO Influence Scores')
plt.show()
