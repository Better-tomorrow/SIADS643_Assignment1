"""Module containing accuracy plot generating function"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def accuracy_plot(knn, x_train:pd.DataFrame, x_test:pd.DataFrame,
                  y_train:pd.Series, y_test:pd.Series) -> None:
    """
    Create a accuracy plot of the K nearest neighbors classifier by target value
    (i.e. malignant, benign).
    
    Args:
        knn (knn Classifier) : Trained K nearest neighbors calssifier
        x_train (pd.DataFrame): training feature data
        y_train (pd.Series): training target
        x_test (pd.DataFrame): testing feature data
        y_test (pd.Series): testing target
 
    Returns:
        None
    """

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_x = x_train[y_train == 0]
    mal_train_y = y_train[y_train == 0]
    ben_train_x = x_train[y_train == 1]
    ben_train_y = y_train[y_train == 1]

    mal_test_x = x_test[y_test == 0]
    mal_test_y = y_test[y_test == 0]
    ben_test_x = x_test[y_test == 1]
    ben_test_y = y_test[y_test == 1]

    scores = [
        knn.score(mal_train_x, mal_train_y),
        knn.score(ben_train_x, ben_train_y),
        knn.score(mal_test_x, mal_test_y),
        knn.score(ben_test_x, ben_test_y),
    ]

    plt.figure(figsize=(8, 6))

    # Plot the scores as a bar chart
    bars = plt.bar(
        np.arange(4), scores, color=["#4c72b0", "#4c72b0", "#55a868", "#55a868"]
    )

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(
            bar.get_x() + bar.get_width() / 2,
            height * 0.90,
            "{0:.{1}f}".format(height, 2),
            ha="center",
            color="w",
            fontsize=11,
        )

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(
        top="off",
        bottom="off",
        left="off",
        right="off",
        labelleft="off",
        labelbottom="on",
    )

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks(
        [0, 1, 2, 3],
        ["Malignant\nTraining", "Benign\nTraining", "Malignant\nTest", "Benign\nTest"],
        alpha=0.8,
    )

    plt.title("Training and Test Accuracies for Malignant and Benign Cells", alpha=0.8)
    plt.show()
