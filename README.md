Machine Learning Model Training and Prediction
==============================

This repository contains Python scripts for training and predicting with various machine learning models, specifically designed for classification and regression tasks. The models include CatBoost, AutoGluon, and RandomForest, and the tasks are organized into separate classes for ease of use.

# Table of Contents

    Directory Structure
    Installation
    Usage
        Command-line Interface
        Choosing a Classifier
        Configuration
    Models
        CatBoost Classifier
        CatBoost Kaggle Classifier
        AutoGluon Classifier Trainer
        RandomForest Classifier Trainer
        AutoGluon Regressor
    Data Paths
    Results
    Additional Notes
    License


# Directory Structure
 
The project has the following directory structure:

.
├── data/
│   ├── classification/
│   │   ├── train_features.csv
│   │   ├── train_label.csv
│   │   └── test_features.csv
│   └── regression/
│       ├── train_features.csv
│       ├── train_label.csv
│       └── test_features.csv
├── models/
├── results/
│   ├── classification/
│   │   ├── prediction_autogluon.csv
│   │   ├── prediction_catboost.csv
│   │   ├── prediction_catboost_kaggle.csv
│   │   └── prediction_rf.csv
│   └── regression/
│       └── prediction_autogluon_regression.csv
├── src/
│   └── models/
│       ├── classification_task.py
│       └── regression_task.py
├── main.py
├── README.md
└── requirements.txt

    data/: Contains datasets for both classification and regression tasks.
    models/: Placeholder for storing trained models.
    results/: Store the predictions for both classification and regression tasks.
    src/models/: Implementation of machine learning tasks.
    main.py: Main script to run different machine learning tasks.
    README.md: This file explaining the project and its usage.
    requirements.txt: Dependencies required to run the project.

# Installation

    Clone the repository:

```bash
git clone https://github.com/your-username/machine-learning-project.git
cd machine-learning-project
```

Install the required dependencies:

```bash
    pip install -r requirements.txt
```

# Usage
1. Command-line Interface

The main entry point for the project is main.py, which provides a command-line interface to run different classifiers. You can use the --classifier argument to choose between classifiers:

```bash
python main.py --classifier catboosthyp
```

2. Choosing a Classifier

    catboosthyp: CatBoost Classifier for hyperparameter tuning.
    catboostkag: CatBoost Kaggle Classifier.
    autogluontrain: AutoGluon Classifier Trainer.
    randomforest: RandomForest Classifier Trainer.
    AutoGluonregression: AutoGluon Regressor for regression task.

3. Configuration

Edit the COMMON_PATHS and CLASS_COMMON_PATHS dictionaries in main.py to adjust file paths for your specific data locations.


# Models
1. CatBoost Classifier

    Trains a CatBoost classifier for a classification task.
    Saves the trained model and predicts on test data.

2. CatBoost Kaggle Classifier

    Specifically designed for Kaggle competition format.
    Uses a CatBoost classifier, handles missing values, and normalizes features.
    Saves the trained model and predictions in Kaggle competition format.

3. AutoGluon Classifier Trainer

    Uses AutoGluon for classification tasks.
    Trains and predicts with AutoGluon.

4. RandomForest Classifier Trainer

    Trains a RandomForest classifier for a classification task.
    Saves the trained model and predicts on test data.

5. AutoGluon Regressor

    Uses AutoGluon for regression tasks.
    Trains and predicts with AutoGluon.

# Data Paths

Adjust the paths in the COMMON_PATHS and CLASS_COMMON_PATHS dictionaries in main.py to match the location of your dataset files.

# Results

The predictions for each classifier will be stored in the results/ directory.

# Additional Notes

    Each classifier class is implemented in separate Python files under src/models/.
    Make sure to have the required libraries installed as specified in requirements.txt.
    Please refer to the specific comments in the code for detailed explanations of each section.

# License

This project is licensed under the MIT License.




ML_Competition
==============================

A project comprising of 2 tasks: Classification and Regression tasks aim at preforming predictions using the test data with the best accuracy or rmse. This project contains scripts for training and evaluating machine learning models using CatBoost, AutoGluon and RandomforestRegressor classifiers for classification tasks with feature reduction.

# Machine Learning Project

 The project is organized into the following directories:

- `code/src`: Contains the source code for the CatBoost and AutoGluon classifiers.
- `code/train`: Contains the main scripts for running the CatBoost and AutoGluon classifiers.
- `code/data`: Should contain the input data files (`train_features.csv`, `train_label.csv`, `test_features.csv`).
- `code/trained_data`: Will store the trained model files (`trained_model_catboost.pkl`, `trained_model_ag.pkl`).
- `code/results`: Will store the prediction results.

## Running the Code

To run the code, follow these steps:

1. Install the required packages using the provided `requirements.txt` file.
2. Navigate to the `code/train` directory.
3. Run the main script with the desired classifier: CatBoostClassifierKaggle, CatBoostClassifierTrainer, AutoGluonClassifier, and RandomForestClassifierTrainer based on the provided command-line argument below

```bash
python main.py --classifier catboost
python main.py --classifier catboost
python main.py --classifier catboost
python main.py --classifier catboost
```


Project Organization
------------

## Project Structure

- **code/src/classification_task.py:** Contains the `AutoGluonClassifier`,  `CatBoostClassifier` and `AutoGluonClassifier` classes, which encapsulates the functionality for training and saving the AutoGluon CatBoost, and RandomforestRegressor classifiers.

- **code/train/main.py:** The main script to run the AutoGluon classification task. It initializes the `AutoGluonClassifier`, reads data, trains the classifier, and saves the results.

- **code/data/:** Directory for storing input data files.

- **code/trained_data/:** Directory for saving the trained AutoGluon model.

- **code/results/:** Directory for saving the classification results.


## Usage
1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run AutoGluon Classifier:
```bash
cd mlcompete
python main.py
python main.py --classifier catboost
python main.py --classifier autogluon
```



    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
