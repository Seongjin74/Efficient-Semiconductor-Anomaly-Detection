# main.py

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Module import
from src.data.loader import load_data
from src.preprocessing.cleaning import (
    dataset_info, remove_columns_with_missing, fill_missing, 
    remove_single_value_columns, remove_time_column, remove_collinear_features, 
    normalize_features
)
from src.preprocessing.visualization import plot_target_distribution
from src.models.classifiers import get_classifiers
from src.evaluation.metrics import evaluate_model, plot_metrics
from src.feature_selection.hybrid import hybrid_feature_selection
from src.pipeline.pipeline import build_hybrid_pipeline

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

def main():
    # Load data
    data = load_data()
    dataset_info(data)
    
    # Data preprocess
    data = remove_columns_with_missing(data, threshold=50)
    data = fill_missing(data)
    data = remove_single_value_columns(data)
    data = remove_time_column(data)
    data = remove_collinear_features(data, threshold=0.7)
    data_scaled = normalize_features(data, target_col="Pass_Fail")
    
    # Visualize target distribution
    plot_target_distribution(data, target_col="Pass_Fail")
    
    # Train-Test Split
    X = data_scaled.drop(columns=["Pass_Fail"])
    y = data_scaled["Pass_Fail"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models
    models = get_classifiers()
    
    # Step 1: Original Imbalanced Data ()
    results_step1 = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc, tpr, fpr = evaluate_model(model, X_test, y_test)
        results_step1.append([name, acc, tpr, fpr])
    results_df_step1 = pd.DataFrame(results_step1, columns=["Model", "Accuracy", "TPR", "FPR"])
    print(results_df_step1)
    plot_metrics(results_df_step1, title="Model Performance (Original Data)")
    
    # Step 2: Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    results_step2 = []
    for name, model in models.items():
        model.fit(X_train_smote, y_train_smote)
        acc, tpr, fpr = evaluate_model(model, X_test, y_test)
        results_step2.append([name, acc, tpr, fpr])
    results_df_step2 = pd.DataFrame(results_step2, columns=["Model", "Accuracy", "TPR", "FPR"])
    print(results_df_step2)
    plot_metrics(results_df_step2, title="Model Performance (After SMOTE)")
    
    # Step 3: SMOTE + PCA
    pca_components = 50
    pca = PCA(n_components=pca_components, random_state=42)
    X_train_smote_pca = pca.fit_transform(X_train_smote)
    X_test_pca = pca.transform(X_test)
    results_step3 = []
    for name, model in models.items():
        model.fit(X_train_smote_pca, y_train_smote)
        acc, tpr, fpr = evaluate_model(model, X_test_pca, y_test)
        results_step3.append([name, acc, tpr, fpr])
    results_df_step3 = pd.DataFrame(results_step3, columns=["Model", "Accuracy", "TPR", "FPR"])
    print(results_df_step3)
    plot_metrics(results_df_step3, title="Model Performance (SMOTE + PCA)")
    
    # Step 4: SMOTE + Hybrid Feature Selection + PCA
    # Apply SMOTE (adjust sampling_strategy)
    smote_fs = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train_smote_fs, y_train_smote_fs = smote_fs.fit_resample(X_train, y_train)
    
    # Hybrid feature selection
    selected_features = hybrid_feature_selection(X_train_smote_fs, y_train_smote_fs, corr_th=0.15, var_th=0.05)
    print("Selected features:", selected_features)
    
    # Build pipeline with GridSearchCV 
    grid_search = build_hybrid_pipeline(selected_features)
    grid_search.fit(X_train_smote_fs, y_train_smote_fs)
    best_estimator = grid_search.best_estimator_
    print("\nBest Estimator")
    print(best_estimator)
    print("\Best Parameters")
    print(grid_search.best_params_)
    
    # Evaluate test data with best pipeline
    y_pred_hybrid = grid_search.predict(X_test)
    from sklearn.metrics import accuracy_score, confusion_matrix
    acc = accuracy_score(y_test, y_pred_hybrid)
    cm = confusion_matrix(y_test, y_pred_hybrid)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    results_hybrid = pd.DataFrame([["Hybrid Model", acc, tpr, fpr]],
                                  columns=["Model", "Accuracy", "TPR", "FPR"])
    print(results_hybrid)
    plot_metrics(results_hybrid, title="Model Performance (Hybrid Pipeline)")
    
if __name__ == "__main__":
    main()
