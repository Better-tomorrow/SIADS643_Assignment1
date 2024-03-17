import pandas as pd

def prepare_X_y(cancer: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate features data and predicted target (type of tumor - Malignant or bening) .
    
    Args:
        cancer (pd.DataFrame) : cancer dataframe
 
    Returns:
        X (pd.DataFrame): features data
        y (pd.Series): target
    """
    
    X, y = None, None

    # Split the dataframe into feature data and target
    X = cancer.drop(labels="target", axis=1)
    y = cancer["target"]
    
    return X, y