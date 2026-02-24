# src/preprocessing.py
"""
Data Preprocessing Module
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    """Handles all data preprocessing tasks"""
    
    def __init__(self, scaling_method='standard'):
        self.scaling_method = scaling_method
        self.scaler = None
        self.imputer = SimpleImputer(strategy='median')
        
    def handle_missing_values(self, df):
        """Handle missing values in dataset"""
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = self.imputer.fit_transform(df_clean[numeric_cols])
        return df_clean
    
    def remove_outliers(self, df, columns, method='iqr', threshold=3):
        """Remove outliers using IQR or Z-score method"""
        df_clean = df.copy()
        
        if method == 'iqr':
            for col in columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        elif method == 'zscore':
            for col in columns:
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())