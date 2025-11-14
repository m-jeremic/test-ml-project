import os
import numpy as np
import pandas as pd

def load_data():
    """
    Load dataset from file
    Returns:
        x (ndarray): Input features
        y (ndarray): Target values
    """
    # get data file directory
    filename = 'temps_data.txt'
    data_path = get_data_dir(filename)

    data = np.loadtxt(filename, data_path, delimiter=',')
    
    x = data[:, 0]
    y = data[:, 1]
    return x, y



def load_data_csv(file_path):
    """
    Loads data from a CSV file.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None


def get_data_dir(filename):
    """
    Get data file directory before loading data
    """
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root and then to the data directory
    data_path = os.path.join(current_dir, '..', 'data', filename)

    return data_path


def extract_features(df, feature_columns):
    """
    Extracts the feature columns from the dataframe.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        feature_columns (list): A list of column names to be used as features.
        
    Returns:
        pd.DataFrame: A dataframe containing only the features.
    """
    return df[feature_columns]


def create_target_variable(df, target_column):
    """
    Creates the target variable (y) from the dataframe.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        target_column (str): The name of the target column.
        
    Returns:
        pd.Series: A series containing the target variable.
    """
    return df[target_column]

