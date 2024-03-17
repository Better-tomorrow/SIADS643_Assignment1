# Breast Cancer Classification using K-Nearest Neighbors

This project uses the K-Nearest Neighbors (KNN) algorithm to classify breast cancer tumors into malignant or benign categories. The dataset used is the Breast Cancer Wisconsin (Diagnostic) dataset provided by the scikit-learn library.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Clone the github repository https://github.com/Better-tomorrow/SIADS643_Assignment1.git
You need to have Python installed on your machine. 
This project uses the following Python libraries:

- numpy
- pandas
- scikit-learn
- matplotlib

You can install these packages using pip:

``` bash
pip install numpy pandas scikit-learn matplotlib
```

## Run the script
To run the script, navigate to the project directory and run the following command:
python analysis_breast_cancer_data.py

## Description of the files
 - analysis_breast_cancer_data.py -> Contains main analysis of the data. All functions are referred here.
 - load_breast_cancer_data.py -> Has make_cancer_dataframe() function which loads the scikit-learn breast cancer data into a pandas dataframe.
 - feature_count.py -> feature_count() function retruns the number of features used for classification prediction task
 - get_distibution.py -> get_target_distro() function returns a histogram of two target values 'malignant' or 'benign' in cancer dataset
 - prepare_X_y.py -> prepare_X_y() function prepares feature dataframe and target seires
 - train_test_split.py -> get_train_test() function splits the data into training and test sets
 - k_nearest_neighbors.py -> k_nearest_neighbors() trains a KNN model using training data
 - generate_accuracy_plot.py -> accuracy_plot() plots the prediction accuracy ay each classification level

## Authors
Deepak V Prabhu
deepakpr@umich.edu