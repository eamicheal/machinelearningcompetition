# -*- coding: utf-8 -*-

"""
Classification Task

Author: Aniebiet Micheal Ezekiel
Date: January 27, 2024

This script contains the implementation of a machine learning project with various classifiers,
including CatBoost, AutoGluon, RandomForest, XGBoost, and others for classification tasks

Note: Make sure to install the required libraries before running the script.
"""

# Standard library imports
import os
import pandas as pd

# AutoGluon imports
from autogluon.tabular import TabularPredictor
from autogluon.multimodal.presets import get_automm_presets
from autogluon.multimodal import MultiModalPredictor

# display imports
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from skopt import BayesSearchCV

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE

# CatBoost imports
from catboost import CatBoostClassifier

# XGBoost imports
import xgboost as xgb

# Joblib for model persistence
import joblib


# Class to handle training and prediction with AutoGluon for a classification task
class AutoGluonClassifierTrainer:
    def __init__(self, train_file_path, train_label_file_path, test_file_path, model_save_path, result_save_path):
        """
        Initialize the AutoGluonClassifierTrainer with file paths and save paths.

        Args:
        - train_file_path (str): Path to the training data CSV file.
        - train_label_file_path (str): Path to the training labels CSV file.
        - test_file_path (str): Path to the test data CSV file.
        - model_save_path (str): Path to save the trained model.
        - result_save_path (str): Path to save the prediction results.
        """
        self.train_file_path = train_file_path
        self.train_label_file_path = train_label_file_path
        self.test_file_path = test_file_path
        self.model_save_path = model_save_path
        self.result_save_path = result_save_path
        self.label_column = 'label'
        self.features_to_remove = ['feature_2', 'feature_12', 'feature_22']
        self.predictor = None
        self.train_data = None  # Added to store training data

    def read_data(self):
        """
        Read and prepare the training data and labels.

        Returns:
        - X (DataFrame): Features of the training data.
        - y (Series): Labels of the training data.
        """
        train_data = pd.read_csv(self.train_file_path)
        train_labels = pd.read_csv(self.train_label_file_path)

        self.train_data = self.prepare_data(train_data, train_labels)
        X = self.train_data.drop(self.label_column, axis=1)
        y = self.train_data[self.label_column]

        return X, y

    def prepare_data(self, data, labels=None):
        """
        Prepare the data by merging features and labels for training data.

        Args:
        - data (DataFrame): Input data.
        - labels (DataFrame, optional): Labels to be merged with the data.

        Returns:
        - data (DataFrame): Prepared data with features and labels.
        """
        if labels is not None:
            data = data.drop(self.features_to_remove, axis=1)
            data['label'] = labels['label']
        return data

    def train_and_predict(self, X, y):
        """
        Train AutoGluon classifier and make predictions on the test data.

        Args:
        - X (DataFrame): Features of the training data.
        - y (Series): Labels of the training data.
        """
        self.train_autogluon_classifier(X, y)
        self.predict_and_save_results()

    def train_autogluon_classifier(self, X, y):
        """
        Train AutoGluon classifier using default hyperparameters.

        Args:
        - X (DataFrame): Features of the training data.
        - y (Series): Labels of the training data.
        """
        hyperparameters, hyperparameter_tune_kwargs = self.get_automm_presets()

        self.predictor = TabularPredictor(
            label=self.label_column, eval_metric='f1', path=self.model_save_path
        ).fit(self.train_data, hyperparameters='default', presets="best_quality", time_limit=3600)

    def get_automm_presets(self):
        """
        Retrieve AutoGluon multimodal presets for hyperparameters.

        Returns:
        - hyperparameters (dict): Hyperparameters for AutoGluon model.
        - hyperparameter_tune_kwargs (dict): Hyperparameter tuning settings.
        """
        from autogluon.multimodal.presets import get_automm_presets
        return get_automm_presets(problem_type="default", presets="high_quality_hpo")

    def predict_and_save_results(self):
        """
        Make predictions on the test data and save results to a CSV file.
        """
        test_features = pd.read_csv(self.test_file_path)
        test_features = self.prepare_data(test_features)

        predictions_ag = self.predictor.predict(test_features)
        result_df = pd.DataFrame({'Id': test_features['Id'], 'Predicted_Label': predictions_ag})

        result_df.to_csv(self.result_save_path, index=False)
        print(result_df)


# Class to handle training and prediction with CatBoost for a classification task
class CatBoostClassifier:
    def __init__(self, train_file_path, train_label_file_path, test_file_path, model_save_path, result_save_path):
        """
        Initialize the CatBoostClassifier with file paths and save paths.

        Args:
        - train_file_path (str): Path to the training data CSV file.
        - train_label_file_path (str): Path to the training labels CSV file.
        - test_file_path (str): Path to the test data CSV file.
        - model_save_path (str): Path to save the trained model.
        - result_save_path (str): Path to save the prediction results.
        """
        self.train_file_path = train_file_path
        self.train_label_file_path = train_label_file_path
        self.test_file_path = test_file_path
        self.model_save_path = model_save_path
        self.result_save_path = result_save_path
        self.label_column = 'label'

    def read_and_preprocess_data(self):
        """
        Read and preprocess the training data for CatBoost training.

        Returns:
        - X_train (DataFrame): Scaled features of the training data.
        - X_val (DataFrame): Scaled features of the validation data.
        - y_train (Series): Labels of the training data.
        - y_val (Series): Labels of the validation data.
        - scaler (StandardScaler): Scaler used for feature scaling.
        """
        # Read training dataset and prepare data for training
        train_data = pd.read_csv(self.train_file_path)
        train_labels = pd.read_csv(self.train_label_file_path)

        # Handle missing values
        train_data = train_data.fillna(train_data.mean())

        # Define features to remove
        features_to_remove = ['feature_2']

        # Remove specified features
        train_data = train_data.drop(features_to_remove, axis=1)

        # Combine features and labels
        train_data['label'] = train_labels['label']

        # Split into features and target variable
        X = train_data.drop('label', axis=1)
        y = train_data['label']

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Split the resampled dataset into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        return X_train, X_val, y_train, y_val, scaler

    def train_catboost_classifier(self, X_train, y_train):
        """
        Train CatBoost classifier with hyperparameter tuning.

        Args:
        - X_train (DataFrame): Scaled features of the training data.
        - y_train (Series): Labels of the training data.

        Returns:
        - best_model_catboost (CatBoostClassifier): Best trained CatBoost model.
        """
        # Define hyperparameter search space for CatBoost
        param_space_catboost = {
            'iterations': (10, 100),
            'depth': (1, 10),
            'learning_rate': (0.01, 1.0, 'log-uniform'),
            'l2_leaf_reg': (0.1, 10.0, 'log-uniform')
        }

        # Use BayesSearchCV for hyperparameter tuning with CatBoost
        cv_catboost = BayesSearchCV(
            CatBoostClassifier(random_state=42, verbose=0),
            param_space_catboost,
            n_iter=30,
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
            n_jobs=-1
        )

        # Fit the model with hyperparameter tuning
        cv_catboost.fit(X_train, y_train)

        # Get the best model
        best_model_catboost = cv_catboost.best_estimator_

        return best_model_catboost

    def evaluate_and_save_results(self, best_model_catboost, X_val, y_val, scaler):
        """
        Evaluate the best CatBoost model on the validation set and save results.

        Args:
        - best_model_catboost (CatBoostClassifier): Best trained CatBoost model.
        - X_val (DataFrame): Scaled features of the validation data.
        - y_val (Series): Labels of the validation data.
        - scaler (StandardScaler): Scaler used for feature scaling.
        """
        # Evaluate the best model on the validation set
        y_val_pred_catboost = best_model_catboost.predict(X_val)

        print("CatBoost Classifier - Classification Report on Validation Set:")
        print(classification_report(y_val, y_val_pred_catboost))

        # Save the trained CatBoostClassifier model
        joblib.dump(best_model_catboost, self.model_save_path)

        # Load test features
        test_features = pd.read_csv(self.test_file_path)

        # Handle missing values in the test set
        test_features = test_features.fillna(test_features.mean())

        # Remove specified features from the test set
        test_features = test_features.drop(['feature_2'], axis=1)

        # Normalize test features using the same MinMaxScaler
        test_features_scaled = scaler.transform(test_features)

        # Use the trained CatBoostClassifier for prediction
        predictions_catboost = best_model_catboost.predict(test_features_scaled)

        # Save predictions to a CSV file
        result_df_catboost = pd.DataFrame({'Id': test_features['Id'], 'Predicted_Label_CatBoost': predictions_catboost})
        result_df_catboost.to_csv(self.result_save_path, index=False)

        # Print or further process the results as needed
        print(result_df_catboost)


from catboost import CatBoostClassifier  ## import CatBoostClassifier

# Class to handle training, evaluation, and prediction with CatBoost for the main classification task
class CatBoostClassifierTrainer:
    def __init__(self, train_file_path, train_label_file_path, test_file_path, model_save_path, result_save_path):
        """
        Initialize CatBoostClassifierTrainer instance.

        Parameters:
        - train_file_path (str): Path to the training features CSV file.
        - train_label_file_path (str): Path to the training labels CSV file.
        - test_file_path (str): Path to the test features CSV file.
        - model_save_path (str): Path to save the trained CatBoost model.
        - result_save_path (str): Path to save the prediction results.
        """
        self.train_file_path = train_file_path
        self.train_label_file_path = train_label_file_path
        self.test_file_path = test_file_path
        self.model_save_path = model_save_path
        self.result_save_path = result_save_path
        self.features_to_remove = ['feature_2']  # based on data analysis, has no correlation
        self.label_column = 'label'
        self.scaler = StandardScaler()
        self.catboost_classifier = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1,
                                                      l2_leaf_reg=1.0, random_state=42, verbose=0)

    def read_and_prepare_data(self):
        """
        Read and prepare training data.

        Returns:
        - X_train, X_val, y_train, y_val: Prepared training and validation sets.
        """
        # Read training dataset
        train_data = pd.read_csv(self.train_file_path)
        train_labels = pd.read_csv(self.train_label_file_path)

        # Handle missing values
        missing_values = train_data.isna().sum()
        if missing_values.any():
            print("Missing Values in Each Column:")
            print(missing_values)
        else:
            print("No Missing Values")

        # Remove specified features
        train_data = train_data.drop(self.features_to_remove, axis=1)

        # Combine features and labels
        train_data[self.label_column] = train_labels[self.label_column]

        # Split into features and target variable
        X = train_data.drop(self.label_column, axis=1)
        y = train_data[self.label_column]

        # Feature scaling using Standard Scaling
        X_scaled = self.scaler.fit_transform(X)

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Split the resampled dataset into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        return X_train, X_val, y_train, y_val

    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        """
        Train CatBoost classifier and evaluate on the validation set.

        Parameters:
        - X_train, y_train: Training set features and labels.
        - X_val, y_val: Validation set features and labels.
        """
        # Train a CatBoostClassifier
        self.catboost_classifier.fit(X_train, y_train)

        # Evaluate the model on the validation set
        y_val_pred = self.catboost_classifier.predict(X_val)
        print("Classification Report on Validation Set:")
        print(classification_report(y_val, y_val_pred))

        # Save the trained model
        joblib.dump(self.catboost_classifier, self.model_save_path)

    def predict_and_save_results(self):
        """
        Load test features, predict using the trained CatBoost model, and save results to a CSV file.
        """
        # Load test features
        test_features = pd.read_csv(self.test_file_path)

        # Handle missing values in the test set
        missing_values_test = test_features.isna().sum()
        if missing_values_test.any():
            print("Missing Values in Test Set:")
            print(missing_values_test)
        else:
            print("No Missing Values in Test Set")

        # Remove specified features from the test set
        test_features = test_features.drop(self.features_to_remove, axis=1)

        # Normalize test features using Standard Scaling
        test_features_scaled = self.scaler.transform(test_features)

        # Use the trained CatBoostClassifier for prediction
        predictions_catboost = self.catboost_classifier.predict(test_features_scaled)

        # Save predictions to a CSV file
        result_df_catboost = pd.DataFrame({'Id': test_features['Id'], 'Predicted_Label_CatBoost': predictions_catboost})
        result_df_catboost.to_csv(self.result_save_path, index=False)

        # Print or further process the results as needed
        print(result_df_catboost)

    def plot_correlation_matrix(self, features, save_path=None):
        """
        Plot and save the correlation matrix of features.

        Parameters:
        - features (pd.DataFrame): DataFrame containing features.
        - save_path (str): Path to save the correlation matrix plot.
        """
        plt.figure(figsize=(16, 12))
        correlation_matrix = features.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix of Features')

        # Use default path if not provided
        if save_path is None:
            save_path = os.path.abspath("../../reports/figure/correlation_matrix.png")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path)
        plt.close()

    def plot_feature_histograms(self, merged_df, save_path=None):
        """
        Plot and save histograms for each feature.

        Parameters:
        - merged_df (pd.DataFrame): DataFrame containing features.
        - save_path (str): Path to save the histograms plot.
        """
        # Use default path if not provided
        if save_path is None:
            save_path = os.path.abspath("../../reports/figure/histograms.png")

        # Assuming the `merged_df` is a DataFrame with 30 features
        data1 = merged_df.iloc[:, 3:33]  # First 30 features

        # Data scaling using StandardScaler
        scaler = StandardScaler()
        data1_scaled = pd.DataFrame(scaler.fit_transform(data1), columns=data1.columns)

        # Plot histograms for each feature
        fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(15, 10))
        fig.subplots_adjust(hspace=0.5)
        for i, ax in enumerate(axes.flatten()):
            if i < data1_scaled.shape[1]:
                ax.hist(data1_scaled.iloc[:, i], bins=20, color='skyblue', edgecolor='black')
                ax.set_title(f'Feature {i + 1}')

        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path)
        plt.close()

# Class to handle training, evaluation, and prediction with CatBoost for the main classification task
class CatBoostClassifierTrainerUSE:
    def __init__(self, train_file_path, train_label_file_path, test_file_path, model_save_path, result_save_path):
        """
        Initialize CatBoostClassifierTrainer instance.

        Parameters:
        - train_file_path (str): Path to the training features CSV file.
        - train_label_file_path (str): Path to the training labels CSV file.
        - test_file_path (str): Path to the test features CSV file.
        - model_save_path (str): Path to save the trained CatBoost model.
        - result_save_path (str): Path to save the prediction results.
        """
        self.train_file_path = train_file_path
        self.train_label_file_path = train_label_file_path
        self.test_file_path = test_file_path
        self.model_save_path = model_save_path
        self.result_save_path = result_save_path
        self.features_to_remove = ['feature_2']  # based on data analysis, has no correlation
        self.label_column = 'label'
        self.scaler = StandardScaler()
        self.catboost_classifier = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1,
                                                      l2_leaf_reg=1.0, random_state=42, verbose=0)

    def read_and_prepare_data(self):
        """
        Read and prepare training data.

        Returns:
        - X_train, X_val, y_train, y_val: Prepared training and validation sets.
        """
        # Read training dataset
        train_data = pd.read_csv(self.train_file_path)
        train_labels = pd.read_csv(self.train_label_file_path)

        # Handle missing values
        train_data = train_data.fillna(train_data.mean())

        # Remove specified features
        train_data = train_data.drop(self.features_to_remove, axis=1)

        # Combine features and labels
        train_data[self.label_column] = train_labels[self.label_column]

        # Split into features and target variable
        X = train_data.drop(self.label_column, axis=1)
        y = train_data[self.label_column]

        # Feature scaling using Standard Scaling
        X_scaled = self.scaler.fit_transform(X)

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Split the resampled dataset into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        return X_train, X_val, y_train, y_val

    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        """
        Train CatBoost classifier and evaluate on the validation set.

        Parameters:
        - X_train, y_train: Training set features and labels.
        - X_val, y_val: Validation set features and labels.
        """
        # Train a CatBoostClassifier
        self.catboost_classifier.fit(X_train, y_train)

        # Evaluate the model on the validation set
        y_val_pred = self.catboost_classifier.predict(X_val)
        print("Classification Report on Validation Set:")
        print(classification_report(y_val, y_val_pred))

        # Save the trained model
        joblib.dump(self.catboost_classifier, self.model_save_path)

    def predict_and_save_results(self):
        """
        Load test features, predict using the trained CatBoost model, and save results to a CSV file.
        """
        # Load test features
        test_features = pd.read_csv(self.test_file_path)

        # Handle missing values in the test set
        test_features = test_features.fillna(test_features.mean())

        # Remove specified features from the test set
        test_features = test_features.drop(self.features_to_remove, axis=1)

        # Normalize test features using Standard Scaling
        test_features_scaled = self.scaler.transform(test_features)

        # Use the trained CatBoostClassifier for prediction
        predictions_catboost = self.catboost_classifier.predict(test_features_scaled)

        # Save predictions to a CSV file
        result_df_catboost = pd.DataFrame({'Id': test_features['Id'], 'Predicted_Label_CatBoost': predictions_catboost})
        result_df_catboost.to_csv(self.result_save_path, index=False)

        # Print or further process the results as needed
        print(result_df_catboost)


# Class to handle training, evaluation, and prediction with CatBoost (Depreciated)
class CatBoostClassifierTrainer_Dummy:
    def __init__(self, train_file_path, train_label_file_path, test_file_path, model_save_path, result_save_path):
        """
        Initialize the CatBoostClassifierTrainer with file paths and model parameters.

        Args:
        - train_file_path (str): Path to the training data CSV file.
        - train_label_file_path (str): Path to the training labels CSV file.
        - test_file_path (str): Path to the test data CSV file.
        - model_save_path (str): Path to save the trained model.
        - result_save_path (str): Path to save the prediction results.
        """
        self.train_file_path = train_file_path
        self.train_label_file_path = train_label_file_path
        self.test_file_path = test_file_path
        self.model_save_path = model_save_path
        self.result_save_path = result_save_path
        self.features_to_remove = []  # List of features to remove from the dataset
        self.label_column = 'label'
        self.scaler = StandardScaler()
        self.catboost_classifier = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1,
                                                      l2_leaf_reg=1.0, random_state=42, verbose=0)

    def read_and_prepare_data(self):
        """
        Read and prepare the training data for CatBoost training.

        Returns:
        - X_train (DataFrame): Scaled features of the training data.
        - X_val (DataFrame): Scaled features of the validation data.
        - y_train (Series): Labels of the training data.
        - y_val (Series): Labels of the validation data.
        """
        # Read training dataset
        train_data = pd.read_csv(self.train_file_path)
        train_labels = pd.read_csv(self.train_label_file_path)

        # Handle missing values
        train_data = train_data.fillna(train_data.mean())

        # Remove specified features
        train_data = train_data.drop(self.features_to_remove, axis=1)

        # Combine features and labels
        train_data[self.label_column] = train_labels[self.label_column]

        # Split into features and target variable
        X = train_data.drop(self.label_column, axis=1)
        y = train_data[self.label_column]

        # Feature scaling using Standard Scaling
        X_scaled = self.scaler.fit_transform(X)

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Split the resampled dataset into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        return X_train, X_val, y_train, y_val

    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        """
        Train the CatBoostClassifier and evaluate its performance.

        Args:
        - X_train (DataFrame): Scaled features of the training data.
        - y_train (Series): Labels of the training data.
        - X_val (DataFrame): Scaled features of the validation data.
        - y_val (Series): Labels of the validation data.
        """
        # Train a CatBoostClassifier
        self.catboost_classifier.fit(X_train, y_train)

        # Evaluate the model on the validation set
        y_val_pred = self.catboost_classifier.predict(X_val)
        print("Classification Report on Validation Set:")
        print(classification_report(y_val, y_val_pred))

        # Save the trained model
        joblib.dump(self.catboost_classifier, self.model_save_path)

    def predict_and_save_results(self):
        """
        Load test features, preprocess data, and save predictions to a CSV file.
        """
        # Load test features
        test_features = pd.read_csv(self.test_file_path)

        # Handle missing values in the test set
        test_features = test_features.fillna(test_features.mean())

        # Remove specified features from the test set
        test_features = test_features.drop(self.features_to_remove, axis=1)

        # Normalize test features using Standard Scaling
        test_features_scaled = self.scaler.transform(test_features)

        # Use the trained CatBoostClassifier for prediction
        predictions_catboost = self.catboost_classifier.predict(test_features_scaled)

        # Save predictions to a CSV file
        result_df_catboost = pd.DataFrame({'Id': test_features['Id'], 'Predicted_Label_CatBoost': predictions_catboost})
        result_df_catboost.to_csv(self.result_save_path, index=False)

        # Print or further process the results as needed
        print(result_df_catboost)


# Class to handle training, evaluation, and prediction with RandomForestClassifier
class RandomForestClassifierTrainer:
    def __init__(self, train_file_path, train_label_file_path, test_file_path, model_save_path, result_save_path):
        """
        Initialize the RandomForestClassifierTrainer with file paths and scaler.

        Args:
        - train_file_path (str): Path to the training data CSV file.
        - train_label_file_path (str): Path to the training labels CSV file.
        - test_file_path (str): Path to the test data CSV file.
        - model_save_path (str): Path to save the trained model.
        - result_save_path (str): Path to save the prediction results.
        """
        self.train_file_path = train_file_path
        self.train_label_file_path = train_label_file_path
        self.test_file_path = test_file_path
        self.model_save_path = model_save_path
        self.result_save_path = result_save_path
        self.minmax_scaler = MinMaxScaler()

    def read_and_prepare_data(self):
        """
        Read and prepare the training data for RandomForestClassifier training.

        Returns:
        - X_train (DataFrame): Scaled features of the training data.
        - X_val (DataFrame): Scaled features of the validation data.
        - y_train (Series): Labels of the training data.
        - y_val (Series): Labels of the validation data.
        """
        # Read training dataset
        train_data = pd.read_csv(self.train_file_path)
        train_labels = pd.read_csv(self.train_label_file_path)

        # Handle missing values
        train_data = train_data.fillna(train_data.mean())

        # Define features to remove
        features_to_remove = ['feature_2', 'feature_12']

        # Remove specified features
        train_data = train_data.drop(features_to_remove, axis=1)

        # Combine features and labels
        train_data['label'] = train_labels['label']

        # Specify label column
        label_column = 'label'

        # Split into features and target variable
        X = train_data.drop(label_column, axis=1)
        y = train_data[label_column]

        # Feature scaling using Min-Max Scaling
        X_scaled = self.minmax_scaler.fit_transform(X)

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Split the resampled dataset into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        return X_train, X_val, y_train, y_val

    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        """
        Train the RandomForestClassifier and evaluate its performance.

        Args:
        - X_train (DataFrame): Scaled features of the training data.
        - y_train (Series): Labels of the training data.
        - X_val (DataFrame): Scaled features of the validation data.
        - y_val (Series): Labels of the validation data.

        Returns:
        - rf_classifier (RandomForestClassifier): Trained RandomForestClassifier model.
        """
        # Train a RandomForestClassifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)

        # Evaluate the model on the validation set
        y_val_pred = rf_classifier.predict(X_val)
        print("Classification Report on Validation Set:")
        print(classification_report(y_val, y_val_pred))

        return rf_classifier

    def save_model(self, model):
        """
        Save the trained model to a specified file path.

        Args:
        - model: Trained machine learning model.
        """
        # Save the trained model
        joblib.dump(model, self.model_save_path)

    def predict_and_save_results(self, model):
        """
        Load test features, preprocess data, and save predictions to a CSV file.

        Args:
        - model: Trained machine learning model.
        """
        # Load test features
        test_features = pd.read_csv(self.test_file_path)

        # Handle missing values in test set
        test_features = test_features.fillna(test_features.mean())

        # Remove specified features from the test set
        test_features = test_features.drop(['feature_2', 'feature_12'], axis=1)

        # Normalize test features using Min-Max Scaling
        test_features_scaled = self.minmax_scaler.transform(test_features)

        # Use the trained RandomForestClassifier for prediction
        predictions_rf = model.predict(test_features_scaled)

        # Save predictions to a CSV file
        result_df = pd.DataFrame({'Id': test_features['Id'], 'Predicted_Label': predictions_rf})
        result_df.to_csv(self.result_save_path, index=False)

        # Print or further process the results as needed
        print(result_df)
