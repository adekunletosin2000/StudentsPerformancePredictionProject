import streamlit as st
import logging
from src.pipeline.predict_pipeline import CustomData, predictpipeline
import pandas as pd

# Set up logging for debugging purposes
logging.basicConfig(level=logging.DEBUG)

# Function to handle prediction (without threading)
@st.cache
def run_prediction(pred_df):
    try:
        # Call the prediction pipeline
        predict_pipeline = predictpipeline()
        results = predict_pipeline.predict(pred_df)

        # Ensure the result is valid before processing
        if isinstance(results, list) and len(results) > 0:
            logging.debug(f"Prediction results: {results[0]}")
            return results[0]  # Return the prediction result
        else:
            logging.error("Prediction failed or returned invalid result.")
            return "Prediction failed or returned invalid result."
    except Exception as e:
        logging.error(f"Prediction task error: {e}")
        return f"Error during prediction: {e}"

# Streamlit app layout
st.title("Prediction App")
st.markdown("Enter the data for prediction")

# User input for prediction
gender = st.selectbox('Gender', ['Male', 'Female'])
race_ethnicity = st.selectbox('Race/Ethnicity', ['Group A', 'Group B', 'Group C'])
parental_level_of_education = st.selectbox('Parental Level of Education', ['High School', 'Associate\'s Degree', 'Bachelor\'s Degree', 'Master\'s Degree'])
lunch = st.selectbox('Lunch', ['Standard', 'Free/Reduced'])
test_preparation_course = st.selectbox('Test Preparation Course', ['Completed', 'None'])

reading_score = st.number_input('Reading Score', min_value=0, max_value=100)
writing_score = st.number_input('Writing Score', min_value=0, max_value=100)

# Handle the form submission and validation
if st.button("Predict"):
    try:
        # Validate scores
        if not (0 <= reading_score <= 100):
            st.error("Reading score must be between 0 and 100.")
        elif not (0 <= writing_score <= 100):
            st.error("Writing score must be between 0 and 100.")
        else:
            # Prepare the data
            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score
            )

            # Convert to DataFrame for prediction
            pred_df = data.get_data_as_dataframe()

            # Show spinner while the prediction is running
            with st.spinner("Running prediction..."):
                prediction_result = run_prediction(pred_df)

            # Show the result after the spinner
            st.success(f"Prediction result: {prediction_result}")

    except ValueError as e:
        st.error(f"Invalid input: {e}")
    except Exception as e:
        logging.error(f"Error: {e}")
        st.error(f"Error: {e}")
