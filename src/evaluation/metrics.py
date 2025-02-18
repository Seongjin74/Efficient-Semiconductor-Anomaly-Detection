# src/evaluation/metrics.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return acc, tpr, fpr

def plot_metrics(results_df, title, bar_width=0.25):
    import numpy as np
    models = results_df["Model"]
    index = np.arange(len(models))
    plt.figure(figsize=(12,5))
    plt.bar(index, results_df["Accuracy"], bar_width, label="Accuracy")
    plt.bar(index + bar_width, results_df["TPR"], bar_width, label="TPR")
    plt.bar(index + 2*bar_width, results_df["FPR"], bar_width, label="FPR")
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(index + bar_width, models, rotation=30, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
