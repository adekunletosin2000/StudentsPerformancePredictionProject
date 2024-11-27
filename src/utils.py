import sys
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  # Import missing metric

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
    
def evaluate_models(X, y, models):
    try:
        # Correcting the train_test_split assignment
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        report = {}
        
        model_keys = list(models.keys())  # Extract model keys once
        model_values = list(models.values())  # Extract model values once

        for i in range(len(model_values)):
            model = model_values[i]

            model.fit(X_train, y_train)  # Train model

            # Predict on train and test
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate r2 score for both train and test
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_keys[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)
