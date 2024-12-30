import os
import sys
import flask
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            # Ensure train_array and test_array have the correct shape
            # Split features and target variable from train and test arrays
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features in train set
                train_array[:, -1],   # Target variable in train set
                test_array[:, :-1],   # Features in test set
                test_array[:, -1]     # Target variable in test set
            )

            # Check if X_train and y_train have consistent dimensions
            assert X_train.shape[0] == y_train.shape[0], "Mismatch between X_train and y_train"
            assert X_test.shape[0] == y_test.shape[0], "Mismatch between X_test and y_test"

            # Define a dictionary of models to evaluate
            models = {
                "Linear Regressor": LinearRegression(),
                "Lasso": Lasso(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegression": XGBRegressor(),
                #"Cat Boost Regressor": CatBoostRegressor(verbose=False),
                "Ada Boost Regressor": AdaBoostRegressor()
            }

            # Define hyperparameter grid for models
            params = {
                "Linear Regressor": {},
                "Lasso": {'alpha': [0.1, 1, 10, 100]},
                "K-Neighbours Regressor": {'n_neighbors': [3, 5, 10]},
                "Decision Tree Regressor": {'min_samples_split': [2, 5, 10]},
                "Random Forest Regressor": {'n_estimators': [100, 200, 500], 'min_samples_split': [2, 5, 10]},
                "XGBRegression": {'n_estimators': [100, 200, 500], 'learning_rate': [0.01, 0.1, 0.3], 'subsample': [0.8, 0.9, 1.0]},
                #"Cat Boost Regressor": {'iterations': [50, 100, 200], 'learning_rate': [0.01, 0.05, 0.1]},
                "Ada Boost Regressor": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}
            }

            # Call the evaluate_models function, passing the required arguments
            report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            
            # Find the best model based on R2 score
            best_model_score = max(sorted(report.values()))
            best_model_name = list(report.keys())[list(report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info("Best model found based on R2 score")

            # Fit the best model before using it for prediction
            best_model.fit(X_train, y_train)

            # Save the best model
            save_obj(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # Predict and calculate R2 score on the test set
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square  # Return R2 score as the evaluation metric

        except Exception as e:
            raise CustomException(e, sys)
