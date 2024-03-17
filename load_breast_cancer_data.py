import pandas as pd
from sklearn.datasets import load_breast_cancer


def make_cancer_dataframe() -> pd.DataFrame:
    """
    Load the cancer data from the scikit-learn datasets. Add a empty target column. 
    Create and return the dataframe.
    
    Args:
        None
 
    Returns:
        cancer_df(pd.DataFrame): The pandas dataframe of cancer dataset
    """

    # Load the breast_cancer dataset from scikit-learn.
    cancer = load_breast_cancer()
    cancer_df = None
    # Create the breast cancer dataframe 
    cancer_df = pd.DataFrame(cancer["data"], columns=cancer["feature_names"])
    # Add a target column at the end of dataframe
    cancer_df.insert(len(cancer_df.columns), "target", cancer["target"])

    return cancer_df