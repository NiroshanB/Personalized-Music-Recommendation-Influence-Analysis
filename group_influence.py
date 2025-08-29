from data_modelling import *
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def compute_group_influence(X_train, y_train, X_valid, y_valid, group_sizes=None, random_state=42):
    if group_sizes is None:
        group_sizes = np.linspace(0.05, 0.55, 10)

    # Set seed for reproducibility
    random.seed(random_state)

    # Train baseline models
    model_nb_base = GaussianNB()
    model_nb_base.fit(X_train, y_train)
    baseline_auc_nb = roc_auc_score(y_valid, model_nb_base.predict_proba(X_valid)[:, 1])

    model_rf_base = RandomForestClassifier(random_state=random_state)
    model_rf_base.fit(X_train, y_train)
    baseline_auc_rf = roc_auc_score(y_valid, model_rf_base.predict_proba(X_valid)[:, 1])

    model_lr_base = LogisticRegression(max_iter=1000, random_state=random_state)
    model_lr_base.fit(X_train, y_train)
    baseline_auc_lr = roc_auc_score(y_valid, model_lr_base.predict_proba(X_valid)[:, 1])

    # Store influence scores
    influence_scores = {
        "NB": [],
        "RF": [],
        "LR": []
    }

    for size in group_sizes:
        num_samples = int(size * len(X_train))
        indices = random.sample(range(len(X_train)), num_samples)

        # Remove group
        X_train_new = X_train.drop(index=X_train.index[indices])
        y_train_new = y_train.drop(index=y_train.index[indices])

        # Naive Bayes
        model_nb = GaussianNB()
        model_nb.fit(X_train_new, y_train_new)
        auc_nb = roc_auc_score(y_valid, model_nb.predict_proba(X_valid)[:, 1])
        influence_scores["NB"].append(abs(auc_nb - baseline_auc_nb))

        # Random Forest
        model_rf = RandomForestClassifier(random_state=random_state)
        model_rf.fit(X_train_new, y_train_new)
        auc_rf = roc_auc_score(y_valid, model_rf.predict_proba(X_valid)[:, 1])
        influence_scores["RF"].append(abs(auc_rf - baseline_auc_rf))

        # Logistic Regression
        model_lr = LogisticRegression(max_iter=1000, random_state=random_state)
        model_lr.fit(X_train_new, y_train_new)
        auc_lr = roc_auc_score(y_valid, model_lr.predict_proba(X_valid)[:, 1])
        influence_scores["LR"].append(abs(auc_lr - baseline_auc_lr))

    return group_sizes, influence_scores


def plot_group_influence(group_sizes, influence_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(group_sizes * 100, influence_scores["NB"], marker='o', label='Naive Bayes')
    plt.plot(group_sizes * 100, influence_scores["RF"], marker='s', label='Random Forest')
    plt.plot(group_sizes * 100, influence_scores["LR"], marker='^', label='Logistic Regression')
    plt.xlabel("Group Size (%)")
    plt.ylabel("Influence Score (AUROC Δ)")
    plt.title("Group Size vs Influence Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    for i, size in enumerate(group_sizes):
        print(f'Group Size: {int(size * 100)}% - NB: {influence_scores["NB"][i]:.5f}, '
              f'RF: {influence_scores["RF"][i]:.5f}, LR: {influence_scores["LR"][i]:.5f}')

group_sizes, influence_scores = compute_group_influence(X_train, y_train, X_valid, y_valid)
plot_group_influence(group_sizes, influence_scores)
