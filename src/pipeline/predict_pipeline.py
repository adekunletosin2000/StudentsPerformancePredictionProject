import sys
import pandas as pd
import os
import pickle
import logging
from src.exception import CustomException
from src.utils import load_object

# Setting up logging for better tracking of issues
logging.basicConfig(level=logging.DEBUG)

class predictpipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Define paths with os.path.join for cross-platform compatibility
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            # Load model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Log the data before scaling
            logging.debug(f"Features before scaling: {features}")
            data_scaled = preprocessor.transform(features)

            # Log the scaled data
            logging.debug(f"Scaled features: {data_scaled}")

            # Make predictions
            preds = model.predict(data_scaled)

            # Log prediction result
            logging.debug(f"Prediction results: {preds}")

            return preds

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise CustomException(f"Error during prediction: {e}", sys)

class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str, 
                 lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert the dictionary to DataFrame
            df = pd.DataFrame(custom_data_input_dict)
            logging.debug(f"Dataframe created for prediction: {df}")

            return df
        
        except Exception as e:
            logging.error(f"Error while creating dataframe: {e}")
            raise CustomException(f"Error while creating dataframe: {e}", sys)

# Function to load objects (model and preprocessor)
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {e}")
        raise CustomException(f"Error loading object from {file_path}: {e}", sys)
