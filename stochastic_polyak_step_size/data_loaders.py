"""
Dataset Loading Functions
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import requests
import zipfile
import io
from pathlib import Path

def load_boston_housing():
    """
    Load Boston Housing dataset from original source    
    Returns:
        X: Features (numpy array)
        y: Target (numpy array)
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    return data, target

def load_california_housing():
    """
    Load California Housing dataset    
    Returns:
        X: Features (numpy array)
        y: Target (numpy array)
    """
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    return housing.data, housing.target

def load_california_housing():
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    return housing.data, housing.target

def load_diabetes():
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    return diabetes.data, diabetes.target

def load_concrete():
    """Concrete Compressive Strength Dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    df = pd.read_excel(url)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def load_energy_efficiency():
    """Energy Efficiency Dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    df = pd.read_excel(url)
    X = df.iloc[:, :-2].values
    y = df.iloc[:, -2:].values  # Both heating and cooling load
    return X, y

def load_kin8nm():
    """Kin8nm Dataset"""
    url = "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff"
    df = pd.read_csv(url)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def load_naval_propulsion():
    """
    Naval Propulsion Dataset
    Predict gas turbine compressor decay state coefficient
    
    Features: 16 (Lever position, Ship speed, Gas Turbine parameters, etc.)
    Target: Gas Turbine compressor decay state coefficient
    Samples: 11,934
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip"
    
    print("Downloading Naval Propulsion dataset...")
    response = requests.get(url)
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # The zip contains a text file
        with z.open('UCI CBM Dataset/data.txt') as f:
            # Read as space-separated values
            df = pd.read_csv(f, sep=r'\s+', header=None)
    
    # Features: columns 0-15 (16 features)
    # Targets: columns 16-17 (2 targets: compressor decay, turbine decay)
    # We'll predict the first target (compressor decay state coefficient)
    X = df.iloc[:, :16].values
    y = df.iloc[:, 16].values
    
    return X, y

def load_protein_structure():
    """Protein Tertiary Structure Dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv"
    df = pd.read_csv(url)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    return X, y

def load_year_prediction_msd():
    """
    Year Prediction MSD Dataset
    Predict the release year of a song from audio features
    
    Note: This is a large dataset (515,345 samples)
    Returns first 100k samples for faster training
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"
    
    print("Downloading Year Prediction MSD dataset (this may take a while)...")
    response = requests.get(url)
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open('YearPredictionMSD.txt') as f:
            df = pd.read_csv(f, header=None)
    
    # First column is target (year), rest are features
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    
    # Use subset for faster training (optional - remove to use full dataset)
    # X = X[:100000]
    # y = y[:100000]
    
    return X, y


def load_ct_slices():
    """
    CT Slices Dataset
    Predict the location of CT slices on axial axis
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip"
    
    print("Downloading CT Slices dataset...")
    response = requests.get(url)
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # The dataset has two files, we'll use the larger one
        with z.open('slice_localization_data.csv') as f:
            df = pd.read_csv(f)
    
    # Last column is target, rest are features
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values
    
    return X, y


def load_bike_sharing():
    """
    Bike Sharing Dataset
    Predict hourly bike rental demand
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
    
    print("Downloading Bike Sharing dataset...")
    response = requests.get(url)
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Use hourly data for more samples
        with z.open('hour.csv') as f:
            df = pd.read_csv(f)
    
    # Drop non-predictive columns
    # instant: record index
    # dteday: date
    # casual, registered: components of cnt (target)
    drop_cols = ['instant', 'dteday', 'casual', 'registered']
    
    # Target is 'cnt' (total count)
    y = df['cnt'].values
    X = df.drop(columns=drop_cols + ['cnt']).values
    
    return X, y


def load_wine_quality():
    """
    Wine Quality Dataset
    Predict wine quality score (combined red and white wines)
    """
    # Red wine
    url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df_red = pd.read_csv(url_red, sep=';')
    
    # White wine
    url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df_white = pd.read_csv(url_white, sep=';')
    
    # Combine datasets
    df = pd.concat([df_red, df_white], ignore_index=True)
    
    # Target is 'quality', rest are features
    y = df['quality'].values
    X = df.drop(columns=['quality']).values
    
    return X, y


def load_appliances_energy():
    """
    Appliances Energy Prediction Dataset
    Predict appliances energy use in a house
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
    
    print("Downloading Appliances Energy dataset...")
    df = pd.read_csv(url)
    
    # Drop date column and lights (not predicting lights)
    # Target is 'Appliances'
    y = df['Appliances'].values
    X = df.drop(columns=['date', 'Appliances', 'lights']).values
    
    return X, y


def load_power_plant():
    """
    Combined Cycle Power Plant Dataset
    Predict net hourly electrical energy output
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip"
    
    print("Downloading Power Plant dataset...")
    response = requests.get(url)
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Read the xlsx file
        with z.open('CCPP/Folds5x2_pp.xlsx') as f:
            df = pd.read_excel(f)
    
    # Last column is target (PE - net hourly electrical energy output)
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values
    
    return X, y

class DatasetLoader:
    """Unified interface for loading datasets"""
    
    LOADERS = {
        'boston': load_boston_housing,
        'california': load_california_housing,
        'diabetes': load_diabetes,
        'concrete': load_concrete,
        'energy': load_energy_efficiency,
        'kin8nm': load_kin8nm,
        'naval': load_naval_propulsion,
        'protein': load_protein_structure,
        'year_prediction_msd': load_year_prediction_msd,
        'ct_slices': load_ct_slices,
        'bike_sharing': load_bike_sharing,
        'wine_quality': load_wine_quality,
        'appliances_energy': load_appliances_energy,
        'power_plant': load_power_plant,
    }
    
    @classmethod
    def load(cls, dataset_name, val_split=0.2, test_split=0.2, seed=3):
        """
        Load and split dataset        
        Args:
            dataset_name: Name of dataset
            val_split: Validation split ratio
            test_split: Test split ratio
            seed: Random seed            
        Returns:
            Dictionary with train/val/test splits (normalized and raw)
        """
        if dataset_name not in cls.LOADERS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Load data
        X, y = cls.LOADERS[dataset_name]()
        
        # Split data
        np.random.seed(seed)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        # Calculate split points
        test_size = int(len(X) * test_split)
        val_size = int(len(X) * val_split)
        train_size = len(X) - test_size - val_size
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        # Split
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Normalize
        scaler_X = MinMaxScaler()
        scaler_X.fit(X_train)
        X_train_norm = scaler_X.transform(X_train)
        X_val_norm = scaler_X.transform(X_val)
        X_test_norm = scaler_X.transform(X_test)
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'X_train_norm': X_train_norm,
            'X_val_norm': X_val_norm,
            'X_test_norm': X_test_norm,
            'scaler_X': scaler_X
        }