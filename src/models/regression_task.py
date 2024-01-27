# -*- coding: utf-8 -*-

"""
Regression Task

Author: Aniebiet Micheal Ezekiel
Date: January 27, 2024

This script contains the implementation of a machine learning project with various classifiers,
including CatBoost, AutoGluon, RandomForest, XGBoost, and others for regression tasks

Note: Make sure to install the required libraries before running the script.
"""

import os
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
# from sklearn.model_selection import StratifiedKFold, BayesSearchCV
from sklearn.metrics import classification_report

from autogluon.multimodal.presets import get_automm_presets
from autogluon.multimodal import MultiModalPredictor
from sklearn.impute import SimpleImputer
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb
import joblib

from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd


class AutoGluonRegressor:
    def __init__(self, train_file_path, label_file_path, test_file_path, model_save_path, result_save_path):
        self.train_file_path = train_file_path
        self.label_file_path = label_file_path
        self.test_file_path = test_file_path
        self.model_save_path = model_save_path
        self.result_save_path = result_save_path
        self.label_column = 'rating'
        self.predictor = None

    def load_and_preprocess_data(self):
        train_data = TabularDataset(self.train_file_path)
        train_data[self.label_column] = pd.read_csv(self.label_file_path)[self.label_column]
        return train_data

    def train_and_predict(self):
        train_data = self.load_and_preprocess_data()
        self.train_autogluon_regressor(train_data)
        self.predict_and_save_results()

    def train_autogluon_regressor(self, train_data):
        self.predictor = TabularPredictor(
            label=self.label_column, problem_type='regression', eval_metric='root_mean_squared_error', path=self.model_save_path
        ).fit(train_data, presets="best_quality")

    def predict_and_save_results(self):
        test_data = TabularDataset(self.test_file_path)
        predictions = self.predictor.predict(test_data)

        result_df = pd.DataFrame({'Id': test_data['Id'], 'Predicted': predictions})
        result_df.to_csv(self.result_save_path, index=False)
        print(result_df)