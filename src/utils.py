import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  # Import missing metric
from sklearn.model_selection import GridSearchCV

import dill
from src.exception import CustomException

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        # Ensure the train_test_split is correctly applied
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        report = {}
        
        model_keys = list(models.keys())  # Extract model keys once
        model_values = list(models.values())  # Extract model values once

        for i in range(len(model_values)):
            model = model_values[i]
            para = param[model_keys[i]]  # Hyperparameters for the specific model
            
            # Hyperparameter tuning using GridSearchCV
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, scoring='r2')
            gs.fit(X_train, y_train)  # Fit GridSearchCV to the training data

            # Get the best model from GridSearchCV
            best_model = gs.best_estimator_

            # Predict on train and test
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate r2 score for both train and test
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_keys[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)
