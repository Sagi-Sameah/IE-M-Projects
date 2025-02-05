import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg for better support in PyCharm

import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import os

# Load the dataset
data = pd.read_csv('bank.csv')

# Handle categorical data by converting them into numerical values (label encoding)
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
encoder = LabelEncoder()

for col in categorical_columns:
    data[col] = encoder.fit_transform(data[col])

# Helper function to calculate entropy
def calculate_entropy(data):
    label_counts = data['y'].value_counts()
    probabilities = label_counts / len(data)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


# Function to split data based on a feature and threshold
def split_data(data, feature, threshold):
    left_split = data[data[feature] <= threshold]
    right_split = data[data[feature] > threshold]
    return left_split, right_split


# Function to calculate χ² statistic for a split
def chi_squared_test(left_split, right_split, feature, target):
    left_counts = left_split.groupby([feature, target]).size().unstack(fill_value=0)
    right_counts = right_split.groupby([feature, target]).size().unstack(fill_value=0)

    total_counts = left_counts + right_counts
    chi2, p_value = stats.chisquare(f_obs=total_counts.values, f_exp=total_counts.mean(axis=0).values)

    return chi2, p_value


# Function to prune the tree using the χ² test with a more stringent p-value threshold
def prune_tree(left_split, right_split, feature, target):
    chi2, p_value = chi_squared_test(left_split, right_split, feature, target)

    # Use a stricter p-value threshold to retain only highly significant splits
    if p_value[0] >= 0.01:  # More stringent pruning
        return None  # Prune this split
    return feature, p_value[0]


# Function to build a decision tree using multiple features and limiting depth
def build_tree(ratio, max_depth=None, depth=0, min_samples_split=10):
    train_data = data.sample(frac=ratio, random_state=42)
    test_data = data.drop(train_data.index)

    best_feature = None
    best_entropy = float('inf')
    best_threshold = None

    # Stop growing the tree if maximum depth is reached
    if max_depth and depth >= max_depth:
        return None

    # Loop through each feature to find the best one for splitting
    for feature in train_data.columns:
        if feature == 'y':  # Skip the target variable
            continue
        threshold = train_data[feature].median()
        left_split, right_split = split_data(train_data, feature, threshold)

        # Ensure the split meets the minimum sample size condition
        if len(left_split) < min_samples_split or len(right_split) < min_samples_split:
            continue

        entropy = calculate_entropy(left_split) + calculate_entropy(right_split)
        if entropy < best_entropy:
            best_entropy = entropy
            best_feature = feature
            best_threshold = threshold

    left_split, right_split = split_data(train_data, best_feature, best_threshold)
    prune_tree(left_split, right_split, best_feature, 'y')

    # Recursively build the tree with increased depth
    build_tree(ratio, max_depth, depth + 1, min_samples_split)

    return None  # Return a tree structure once fully implemented


# Function to predict if a customer will open a term deposit
def will_open_deposit(row_input):
    if row_input['age'] <= 30:
        if row_input['job'] in [0, 1]:  # Example: jobs like 'admin', 'technician'
            return 0  # Predict 'no'
        else:
            return 1  # Predict 'yes'
    else:
        if row_input['balance'] > 2000:
            return 1  # Predict 'yes'
        else:
            return 0  # Predict 'no'


# Function to calculate the error of the decision tree for a given test set
def calculate_error(tree, test_data):
    correct_predictions = 0
    total_predictions = len(test_data)

    for _, row in test_data.iterrows():
        prediction = will_open_deposit(row)
        if prediction == row['y']:
            correct_predictions += 1

    error_rate = 1 - correct_predictions / total_predictions
    return error_rate


# Function for k-fold cross-validation
def tree_error(k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    errors = []

    # Create directory to save images if it doesn't exist
    if not os.path.exists('decision_trees'):
        os.makedirs('decision_trees')

    # Initialize the classifier for the final tree
    clf = None

    for train_index, test_index in kf.split(data):
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]

        # Train the model using sklearn's DecisionTreeClassifier for graphical output
        X = train_data.drop(columns=['y'])
        y = train_data['y']
        clf = DecisionTreeClassifier(random_state=42, max_depth=5)
        clf.fit(X, y)

        # Calculate the error on the test data
        error = calculate_error(clf, test_data)
        errors.append(error)

    # Save the decision tree plot after all folds
    plt.figure(figsize=(15, 10))  # Increase figure size to make the tree readable
    plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
    plot_filename = 'decision_trees/decision_tree_final.png'  # Save as a single final plot
    plt.savefig(plot_filename)
    print(f"Saved final decision tree to {plot_filename}")
    plt.close()  # Close the plot after saving

    average_error = np.mean(errors)
    print(f"Average error across {k} folds: {average_error}")


# Test the k-fold cross-validation with k = 5
tree_error(5)

from sklearn.tree import _tree

# Function to print the decision tree in a text-based format
def print_text_tree(decision_tree, feature_names, class_names, spacing="  "):
    tree = decision_tree.tree_
    def recurse(node, depth=0):
        indent = spacing * depth
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            print(f"{indent}If {name} <= {threshold:.2f}:")
            recurse(tree.children_left[node], depth + 1)
            print(f"{indent}Else (if {name} > {threshold:.2f}):")
            recurse(tree.children_right[node], depth + 1)
        else:
            value = tree.value[node]
            class_index = value.argmax()
            print(f"{indent}Predict {class_names[class_index]}")

    recurse(0)


# Integrate the text-based printout into your k-fold cross-validation function
def tree_error(k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    errors = []

    # Create directory to save images if it doesn't exist
    if not os.path.exists('decision_trees'):
        os.makedirs('decision_trees')

    clf = None  # Initialize the classifier for the final tree

    for train_index, test_index in kf.split(data):
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]

        # Train the model using sklearn's DecisionTreeClassifier
        X = train_data.drop(columns=['y'])
        y = train_data['y']
        clf = DecisionTreeClassifier(random_state=42, max_depth=5)
        clf.fit(X, y)

        # Calculate the error on the test data
        error = calculate_error(clf, test_data)
        errors.append(error)

    # Save the decision tree plot after all folds
    plt.figure(figsize=(15, 10))  # Increase figure size to make the tree readable
    plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
    plot_filename = 'decision_trees/decision_tree_final.png'
    plt.savefig(plot_filename)
    print(f"Saved final decision tree to {plot_filename}")
    plt.close()  # Close the plot after saving

    # Print the text-based decision tree
    print("\nText-based Decision Tree:")
    print_text_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'])

    average_error = np.mean(errors)
    print(f"Average error across {k} folds: {average_error}")


# Test the k-fold cross-validation with k = 5
tree_error(5)
