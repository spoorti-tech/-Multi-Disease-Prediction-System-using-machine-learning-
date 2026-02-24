# src/data_loader.py
"""
Data Loader Module for Multi-Disease Prediction
Generates synthetic medical data for demonstration
"""

import numpy as np
import pandas as pd
from pathlib import Path

class DataLoader:
    """Handles loading and generation of medical datasets"""
    
    @staticmethod
    def generate_diabetes_data(n_samples=1000, random_state=42):
        """Generate synthetic diabetes dataset"""
        np.random.seed(random_state)
        
        data = {
            'Pregnancies': np.random.randint(0, 10, n_samples),
            'Glucose': np.random.normal(120, 40, n_samples).clip(50, 250),
            'BloodPressure': np.random.normal(70, 15, n_samples).clip(40, 130),
            'SkinThickness': np.random.normal(20, 10, n_samples).clip(5, 60),
            'Insulin': np.random.normal(100, 50, n_samples).clip(15, 400),
            'BMI': np.random.normal(30, 8, n_samples).clip(15, 60),
            'DiabetesPedigreeFunction': np.random.uniform(0.1, 2.0, n_samples),
            'Age': np.random.randint(21, 80, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Generate target based on risk factors
        risk_score = (
            (df['Glucose'] > 140) * 0.3 +
            (df['BMI'] > 30) * 0.25 +
            (df['Age'] > 40) * 0.2 +
            (df['Pregnancies'] > 3) * 0.15 +
            df['DiabetesPedigreeFunction'] * 0.1
        )
        
        df['Outcome'] = (risk_score + np.random.random(n_samples) * 0.2 > 0.5).astype(int)
        return df
    
    @staticmethod
    def generate_heart_data(n_samples=1000, random_state=42):
        """Generate synthetic heart disease dataset"""
        np.random.seed(random_state)
        
        data = {
            'Age': np.random.randint(29, 80, n_samples),
            'Sex': np.random.choice([0, 1], n_samples, p=[0.45, 0.55]),
            'ChestPainType': np.random.choice([0, 1, 2, 3], n_samples, p=[0.25, 0.25, 0.25, 0.25]),
            'RestingBP': np.random.normal(130, 20, n_samples).clip(90, 200),
            'Cholesterol': np.random.normal(200, 50, n_samples).clip(100, 350),
            'FastingBS': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
            'RestingECG': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]),
            'MaxHR': np.random.normal(150, 30, n_samples).clip(70, 210),
            'ExerciseAngina': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'Oldpeak': np.random.normal(1, 1.5, n_samples).clip(-2, 6),
            'ST_Slope': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.35, 0.25])
        }
        
        df = pd.DataFrame(data)
        
        # Generate target
        risk_score = (
            (data['Age'] > 55) * 0.15 +
            (data['Sex'] == 1) * 0.1 +
            (data['ChestPainType'] >= 2) * 0.25 +
            (data['Cholesterol'] > 240) * 0.15 +
            (data['FastingBS'] == 1) * 0.15 +
            (data['MaxHR'] < 130) * 0.1 +
            (data['ExerciseAngina'] == 1) * 0.1
        )
        
        df['HeartDisease'] = (np.array(risk_score) + np.random.random(n_samples) * 0.2 > 0.45).astype(int)
        return df
    
    @staticmethod
    def generate_parkinson_data(n_samples=1000, random_state=42):
        """Generate synthetic Parkinson's disease dataset"""
        np.random.seed(random_state)
        
        # Parkinson's features (simulated voice measurements)
        data = {
            'MDVP:Fo(Hz)': np.random.normal(150, 30, n_samples).clip(80, 260),
            'MDVP:Fhi(Hz)': np.random.normal(200, 50, n_samples).clip(100, 400),
            'MDVP:Flo(Hz)': np.random.normal(100, 25, n_samples).clip(50, 200),
            'MDVP:Jitter(%)': np.random.normal(0.01, 0.005, n_samples).clip(0.002, 0.05),
            'MDVP:Jitter(Abs)': np.random.normal(0.00005, 0.00002, n_samples).clip(0.00001, 0.0002),
            'MDVP:RAP': np.random.normal(0.005, 0.002, n_samples).clip(0.001, 0.02),
            'MDVP:PPQ': np.random.normal(0.005, 0.002, n_samples).clip(0.001, 0.02),
            'Jitter:DDP': np.random.normal(0.015, 0.007, n_samples).clip(0.003, 0.06),
            'MDVP:Shimmer': np.random.normal(0.04, 0.02, n_samples).clip(0.01, 0.15),
            'MDVP:Shimmer(dB)': np.random.normal(0.3, 0.2, n_samples).clip(0.1, 1.5),
            'Shimmer:APQ3': np.random.normal(0.015, 0.008, n_samples).clip(0.005, 0.06),
            'Shimmer:APQ5': np.random.normal(0.02, 0.01, n_samples).clip(0.005, 0.08),
            'Shimmer:APQ': np.random.normal(0.025, 0.012, n_samples).clip(0.005, 0.1),
            'MDVP:Shimmer(dB)': np.random.normal(0.3, 0.2, n_samples).clip(0.1, 1.5),
            'NHR': np.random.normal(0.01, 0.005, n_samples).clip(0.002, 0.04),
            'HNR': np.random.normal(25, 5, n_samples).clip(10, 35),
            'RPDE': np.random.normal(0.4, 0.15, n_samples).clip(0.1, 0.8),
            'DFA': np.random.normal(0.7, 0.1, n_samples).clip(0.5, 1.0),
            'spread1': np.random.normal(-5, 2, n_samples).clip(-8, -1),
            'spread2': np.random.normal(0.2, 0.1, n_samples).clip(0, 0.5),
            'D2': np.random.normal(2.5, 0.5, n_samples).clip(1.5, 3.5),
            'PPE': np.random.normal(0.2, 0.08, n_samples).clip(0.05, 0.5)
        }
        
        df = pd.DataFrame(data)
        
        # Generate target
        risk_score = (
            (data['MDVP:Jitter(%)'] > 0.015) * 0.2 +
            (data['NHR'] > 0.02) * 0.2 +
            (data['HNR'] < 22) * 0.2 +
            (data['RPDE'] > 0.5) * 0.2 +
            (data['PPE'] > 0.25) * 0.2
        )
        
        df['Status'] = (np.array(risk_score) + np.random.random(n_samples) * 0.2 > 0.45).astype(int)
        return df
    
    @staticmethod
    def generate_breast_cancer_data(n_samples=1000, random_state=42):
        """Generate synthetic breast cancer dataset"""
        np.random.seed(random_state)
        
        # Mean features
        data = {
            'Radius_Mean': np.random.normal(14, 4, n_samples).clip(6, 30),
            'Texture_Mean': np.random.normal(19, 4, n_samples).clip(10, 35),
            'Perimeter_Mean': np.random.normal(90, 25, n_samples).clip(40, 180),
            'Area_Mean': np.random.normal(600, 200, n_samples).clip(150, 1500),
            'Smoothness_Mean': np.random.normal(0.1, 0.03, n_samples).clip(0.05, 0.2),
            'Compactness_Mean': np.random.normal(0.1, 0.05, n_samples).clip(0.02, 0.3),
            'Concavity_Mean': np.random.normal(0.1, 0.08, n_samples).clip(0, 0.4),
            'ConcavePoints_Mean': np.random.normal(0.05, 0.04, n_samples).clip(0, 0.2),
            'Symmetry_Mean': np.random.normal(0.18, 0.05, n_samples).clip(0.1, 0.35),
            'FractalDimension_Mean': np.random.normal(0.06, 0.02, n_samples).clip(0.02, 0.12)
        }
        
        df = pd.DataFrame(data)
        
        # Generate target (0=Benign, 1=Malignant)
        risk_score = (
            (data['Radius_Mean'] > 15) * 0.2 +
            (data['Area_Mean'] > 800) * 0.2 +
            (data['Concavity_Mean'] > 0.15) * 0.25 +
            (data['ConcavePoints_Mean'] > 0.08) * 0.25 +
            (data['Texture_Mean'] > 22) * 0.1
        )
        
        df['Diagnosis'] = (np.array(risk_score) + np.random.random(n_samples) * 0.15 > 0.5).astype(int)
        return df
    
    @staticmethod
    def generate_liver_data(n_samples=1000, random_state=42):
        """Generate synthetic liver disease dataset"""
        np.random.seed(random_state)
        
        data = {
            'Age': np.random.randint(18, 80, n_samples),
            'Gender': np.random.choice([0, 1], n_samples, p=[0.45, 0.55]),
            'Total_Bilirubin': np.random.normal(1.2, 0.8, n_samples).clip(0.2, 8),
            'Direct_Bilirubin': np.random.normal(0.5, 0.4, n_samples).clip(0.1, 4),
            'Alkaline_Phosphotase': np.random.normal(100, 40, n_samples).clip(30, 300),
            'Alamine_Aminotransferase': np.random.normal(40, 30, n_samples).clip(5, 200),
            'Aspartate_Aminotransferase': np.random.normal(45, 35, n_samples).clip(10, 250),
            'Total_Protiens': np.random.normal(6.5, 1, n_samples).clip(3, 9),
            'Albumin': np.random.normal(3.5, 0.6, n_samples).clip(1.5, 5.5),
            'A_G_Ratio': np.random.normal(1.2, 0.3, n_samples).clip(0.5, 2.5)
        }
        
        df = pd.DataFrame(data)
        
        # Generate target
        risk_score = (
            (data['Age'] > 50) * 0.1 +
            (data['Total_Bilirubin'] > 1.5) * 0.2 +
            (data['Alkaline_Phosphotase'] > 140) * 0.2 +
            (data['Alamine_Aminotransferase'] > 60) * 0.2 +
            (data['Aspartate_Aminotransferase'] > 65) * 0.2 +
            (data['Total_Protiens'] < 6) * 0.1
        )
        
        df['Liver_Disease'] = (np.array(risk_score) + np.random.random(n_samples) * 0.2 > 0.5).astype(int)
        return df
    
    @staticmethod
    def load_data(disease_type, n_samples=1000):
        """Load or generate data for specific disease"""
        generators = {
            'diabetes': DataLoader.generate_diabetes_data,
            'heart': DataLoader.generate_heart_data,
            'parkinson': DataLoader.generate_parkinson_data,
            'breast_cancer': DataLoader.generate_breast_cancer_data,
            'liver': DataLoader.generate_liver_data
        }
        
        if disease_type not in generators:
            raise ValueError(f"Unknown disease type: {disease_type}")
        
        return generators[disease_type](n_samples)
    
    @staticmethod
    def save_data(df, disease_type):
        """Save dataset to CSV"""
        Path("data").mkdir(parents=True, exist_ok=True)
        filepath = f"data/{disease_type}_data.csv"
        df.to_csv(filepath, index=False)
        return filepath