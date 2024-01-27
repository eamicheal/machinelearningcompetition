# Project.py - Entry Script
# Author: Aniebiet Micheal Ezekiel
# Date: 2024-01-27
# Description: This script serves as the entry point
#   for the project, utilizing various classification and regression tasks.


# Import necessary libraries and modules
import os
import argparse
from src.models.classification_task import CatBoostClassifierKaggle
from src.models.classification_task import CatBoostClassifier
from src.models.classification_task import AutoGluonClassifierTrainer
from src.models.classification_task import RandomForestClassifierTrainer
from src.models.regression_task import AutoGluonRegressor

# Get the absolute path to the directory containing main.py
file_dir = os.path.dirname(os.path.abspath(__file__))

# Define common file paths for Regression task
COMMON_PATHS = {
    'train_file_path': 'data/regression/train_features.csv',
    'label_file_path': 'data/regression/train_label.csv',
    'test_file_path': 'data/regression/test_features.csv',
    'result_save_path': 'results/regression/',
    'model_save_path_ag_regression': 'models/ag_trained_model_regression.pkl',
    'relative_result_save_path_ag_regression': 'prediction_autogluon_regression.csv',
}

# Define common file paths for Classification task
CLASS_COMMON_PATHS = {
    'train_file_path': 'data/classification/train_features.csv',
    'train_label_file_path': 'data/classification/train_label.csv',
    'test_file_path': 'data/classification/test_features.csv',
    'result_save_path': 'results/classification/',
    'catboost_model_save_path': 'models/trained_model_catboost.pkl',
    'catboost_model_save_path_kaggle': 'models/trained_model_catboost_kaggle.pkl',
    'autogluon_model_save_path': 'models/trained_model_autogluon.pkl',
    'randomforest_model_save_path': 'models/trained_model_rf.pkl',
    'model_save_path_ag': 'models/trained_model_ag.pkl',
    'relative_result_save_path_ag': 'prediction_autogluon.csv',
    'relative_result_save_path_catboost': 'prediction_catboost.csv',
    'relative_result_save_path_catboost_kaggle': 'prediction_catboost_kaggle.csv',
    'relative_result_save_path_rf': 'prediction_rf.csv',
}


def parse_arguments():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run either CatBoostHyper or classification or regression or autogluontrain or RandomForest classifier.")
    parser.add_argument("--classifier",
                        choices=["catboosthyp", "classification", "autogluontrain", "regression",
                                 "randomforest"], required=True, help="Choose classifier.")
    return parser.parse_args()


def main():
    # Entry point of the script
    args = parse_arguments()

    if args.classifier == "catboosthyp":
        run_catboost_classifier(CLASS_COMMON_PATHS)
    elif args.classifier == "classification":
        run_catboost_kaggle(CLASS_COMMON_PATHS)
    elif args.classifier == "autogluontrain":
        run_autogluon_classifier_train(CLASS_COMMON_PATHS)
    elif args.classifier == "randomforest":
        run_randomforest_classifier(CLASS_COMMON_PATHS)
    elif args.classifier == "regression":
        run_autogluon_regressor(COMMON_PATHS)
    else:
        print("Invalid classifier choice. Use --help for available options.")


def run_catboost_kaggle(class_common_paths):
    # Run CatBoost Kaggle task
    train_file_path = class_common_paths['train_file_path']
    train_label_file_path = class_common_paths['train_label_file_path']
    test_file_path = class_common_paths['test_file_path']
    model_save_path = class_common_paths['catboost_model_save_path_kaggle']
    relative_result_save_path = class_common_paths['result_save_path'] + class_common_paths[
        'relative_result_save_path_catboost_kaggle']

    # Create CatBoostTrainer instance
    catboost_trainer = CatBoostClassifierKaggle(train_file_path, train_label_file_path, test_file_path, model_save_path, relative_result_save_path)

    # Read and prepare data
    X_train, X_val, y_train, y_val = catboost_trainer.read_and_prepare_data()

    # Train and evaluate
    catboost_trainer.train_and_evaluate(X_train, y_train, X_val, y_val)

    # Predict and save results
    catboost_trainer.predict_and_save_results()


def run_catboost_classifier(class_common_paths):
    # Run CatBoost Classifier task
    train_file_path = class_common_paths['train_file_path']
    train_label_file_path = class_common_paths['train_label_file_path']
    test_file_path = class_common_paths['test_file_path']
    model_save_path = class_common_paths['catboost_model_save_path']
    relative_result_save_path = class_common_paths['result_save_path'] + class_common_paths[
        'relative_result_save_path_catboost']

    catboost_trainer = CatBoostClassifier(train_file_path, train_label_file_path, test_file_path,
                                          model_save_path, relative_result_save_path)

    # Read and preprocess data
    X_train, X_val, y_train, y_val, scaler = catboost_trainer.read_and_preprocess_data()

    # Train CatBoost classifier
    best_model_catboost = catboost_trainer.train_catboost_classifier(X_train, y_train)

    # Evaluate and save results
    catboost_trainer.evaluate_and_save_results(best_model_catboost, X_val, y_val, scaler)


def run_autogluon_classifier_train(class_common_paths):
    # Run AutoGluon Classifier Trainer task
    ag_classifierTrainer = AutoGluonClassifierTrainer(
        class_common_paths['train_file_path'], class_common_paths['train_label_file_path'],
        class_common_paths['test_file_path'], class_common_paths['model_save_path_ag'],
        class_common_paths['result_save_path'] + class_common_paths['relative_result_save_path_ag']
    )
    X, y = ag_classifierTrainer.read_data()  ## Comments
    ag_classifierTrainer.train_and_predict(X, y)


def run_randomforest_classifier(class_common_paths):
    # Run RandomForest Classifier task
    train_file_path = class_common_paths['train_file_path']
    train_label_file_path = class_common_paths['train_label_file_path']
    test_file_path = class_common_paths['test_file_path']
    model_save_path = class_common_paths['randomforest_model_save_path']
    relative_result_save_path = class_common_paths['result_save_path'] + class_common_paths[
        'relative_result_save_path_rf']

    rf_trainer = RandomForestClassifierTrainer(train_file_path, train_label_file_path, test_file_path,
                                               model_save_path, relative_result_save_path)

    # Read and prepare data
    X_train, X_val, y_train, y_val = rf_trainer.read_and_prepare_data()

    # Train RandomForest classifier
    trained_rf_classifier = rf_trainer.train_and_evaluate(X_train, y_train, X_val, y_val)

    # Save model
    rf_trainer.save_model(trained_rf_classifier)

    # Predict and save results
    rf_trainer.predict_and_save_results(trained_rf_classifier)


def run_autogluon_regressor(common_paths):
    # Run AutoGluon Regressor task
    regressor = AutoGluonRegressor(
        common_paths['train_file_path'], common_paths['label_file_path'],
        common_paths['test_file_path'], common_paths['model_save_path_ag_regression'],
        common_paths['result_save_path'] + common_paths['relative_result_save_path_ag_regression']
    )
    regressor.train_and_predict()


if __name__ == "__main__":
    main()