"""Module containing function for generating training and test data split"""
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def get_train_test(
    x: (pd.DataFrame | np.ndarray), y: (pd.Series | np.ndarray), random_state: int
) -> tuple:
    """
    Generate training and test split of the data.
    
    Args:
        x (pd.DataFrame) : cancer feature dataframe
        y (Pd.Series) : cancer target data
 
    Returns:
        x_train (pd.DataFrame): training feature data
        y_train (pd.Series): training target
        x_test (pd.DataFrame): testing feature data
        y_test (pd.Series): testing target
    """
    x_train, x_test, y_train, y_test = (None,) * 4
    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state)

    return x_train, x_test, y_train, y_test
