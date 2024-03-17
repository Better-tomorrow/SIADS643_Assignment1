from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def k_nearest_neighbors(X_train:pd.DataFrame, y_train:pd.Series) -> KNeighborsClassifier:
    """
    Create and fit k nearest neighbors model on training data.
    
    Args:
        X_train (pd.DataFrame): training feature data
        y_train (pd.Series): training target
 
    Returns:
        knn (KNeighborsClassifier) - > trained model
    """
    knn = None

    # create a model and fir training data
    model = KNeighborsClassifier(n_neighbors=1)
    knn = model.fit(X_train, y_train)
    return knn