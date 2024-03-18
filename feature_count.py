"""Module containing feature count function"""
import pandas as pd

def feature_count(cancer: pd.DataFrame) -> int:
    """
    Calculate the number of features in scikit-learn breast cancer dataset.
    
    Args:
        cancer (dataset) : cancer dataset
 
    Returns:
        result(int): number of features in cancer dataset
    """
    result = None

    # Calculate the count of features in scikit-learn breast cancer dataset
    result_df=cancer.iloc[:,:-1]
    result = len(result_df.columns)
    return result
