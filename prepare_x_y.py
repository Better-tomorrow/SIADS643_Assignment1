"""Module containing function for preparing feature set and target list"""
import pandas as pd

def prepare_x_y(cancer: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate features data and predicted target (type of tumor - Malignant or bening) .
    
    Args:
        cancer (pd.DataFrame) : cancer dataframe
 
    Returns:
        X (pd.DataFrame): features data
        y (pd.Series): target
    """
    x, y = None, None

    # Split the dataframe into feature data and target
    x = cancer.drop(labels="target", axis=1)
    y = cancer["target"]

    return x, y
