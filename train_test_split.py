from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def get_train_test(
    X: (pd.DataFrame | np.ndarray), y: (pd.Series | np.ndarray), random_state: int
) -> tuple:
    X_train, X_test, y_train, y_test = (None,) * 4

    """
    Generate training and test split of the data.
    
    Args:
        X (pd.DataFrame) : cancer feature dataframe
        y (Pd.Series) : cancer target data
 
    Returns:
        X_train (pd.DataFrame): training feature data
        y_train (pd.Series): training target
        X_test (pd.DataFrame): testing feature data
        y_test (pd.Series): testing target
    """

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    
    return X_train, X_test, y_train, y_test