"""Module containing function for generating histogram"""
import numpy as np
import pandas as pd

def get_target_distro(cancer: pd.DataFrame) -> pd.Series:
    """
    Calculate histogram. Frequency of data distribution.
    
    Args:
        cancer (pd.DataFrame) : cancer dataset
 
    Returns:
        distro(pd.Series): histogram distribution
    """
    distro = None

    # generate histogram out of 'target' column. The type of tumor can be 'maligant' or 'benign'.
    hist, _ = np.histogram(cancer["target"],bins=np.linspace(0,1,3))
    distro = pd.Series(hist,index=['malignant', 'benign'])

    return distro
