from load_breast_cancer_data import make_cancer_dataframe
from feature_count import feature_count
from get_distibution import get_target_distro
from prepare_X_y import prepare_X_y
from train_test_split import get_train_test
from k_nearest_neighbors import k_nearest_neighbors
from generate_accuracy_plot import accuracy_plot

def analysis_breast_cancer_data() -> None:
    """
    Analyze Wisconsin breast cancer dataset. Run a K nearest neighbor classifier. Test the accuracy
    of classification model. Plot the prediction accuracy.
    
    Args:
        None
    Returns:
        None
    """
    #Define a random state as value 42 to get the reproducable results
    random_state = 42
    preds = None
    score = None

    # load the breast cancer data from scikit-learn into a dataframe
    cancer_df = make_cancer_dataframe()
    # Prepare feature dataframe and predictor varaible list
    X , y = prepare_X_y(cancer_df)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = get_train_test(X, y, random_state)
    # Generate a trained k nearest neighbor classifier
    knn_classifier = k_nearest_neighbors(X_train, y_train)
    # predict the target on test data using trained k nearest neighbor classifier
    preds = knn_classifier.predict(X_test)
    # Accuracy score of the prediction from calssifier
    score = knn_classifier.score(X_test, y_test)
    # Display the number of features, target class distribution, predicted values and accuracy score
    print("Number of features :",feature_count(cancer_df))
    print("Target class distribution:",get_target_distro(cancer_df))
    print("Predicted values on test split :",preds)
    print("Actual test split target values:",y_test.to_numpy())
    print("Score: ", score)
    # Generate accuracy plot
    accuracy_plot(knn_classifier,X_train,X_test,y_train,y_test)

if __name__ == '__main__':
     analysis_breast_cancer_data()