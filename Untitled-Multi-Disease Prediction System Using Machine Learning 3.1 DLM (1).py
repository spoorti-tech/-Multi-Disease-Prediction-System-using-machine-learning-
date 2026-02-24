# src/__init__.py
"""
Multi-Disease Prediction System
A comprehensive ML system for predicting multiple diseases
"""

__version__ = "1.0.0"
__author__ = "ML Engineer"

from .models import DiseaseModel
from .data_loader import DataLoader
from .utils import load_models, save_models