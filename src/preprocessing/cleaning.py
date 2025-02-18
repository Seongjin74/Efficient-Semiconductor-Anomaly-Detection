# src/preprocessing/cleaning.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def dataset_info(df):
    print(df.shape)
    print(df.info())
    print(df.describe())

def missing_values(df):
    missing = df.isnull().sum()
    missing_percent = (missing / df.shape[0]) * 100
    missing_df = pd.DataFrame({'Missing_Records': missing, 'Percentage': missing_percent})
    return missing_df[missing_df['Missing_Records'] > 0].sort_values(by='Percentage', ascending=False)

def remove_columns_with_missing(df, threshold=50):
    missing_df = missing_values(df)
    columns_to_drop = missing_df[missing_df["Percentage"] > threshold].index
    df = df.drop(columns=columns_to_drop)
    print(df.shape)
    return df

def fill_missing(df):
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

def remove_single_value_columns(df):
    unique_columns = [col for col in df.columns if df[col].nunique() == 1]
    df = df.drop(columns=unique_columns)
    print(df.shape)
    return df

def remove_time_column(df):
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
        print(df.shape)
    return df

def remove_collinear_features(df, threshold=0.7):
    corr_matrix = df.corr()
    drop_cols = set()
    cols = corr_matrix.columns
    for i in range(len(cols)-1):
        for j in range(i+1, len(cols)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                drop_cols.add(cols[j])
    df = df.drop(columns=drop_cols)
    print(df.shape)
    return df

def normalize_features(df, target_col="Pass_Fail"):
    scaler = StandardScaler()
    features = df.drop(columns=[target_col])
    scaled_features = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
    df_scaled[target_col] = df[target_col].values
    print(df_scaled.shape)
    return df_scaled
