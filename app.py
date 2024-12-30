import logging
from flask import Flask, request, render_template, flash
from src.pipeline.predict_pipeline import CustomData, predictpipeline

# Set up logging for debugging purposes
logging.basicConfig(level=logging.DEBUG)

application = Flask(__name__)
application.secret_key = "some_secret_key"  # Secret key for sessions/flash messages

app = application

# Home route for landing page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route for handling form submissions
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoints():
    if request.method == 'GET':
        return render_template('home.html')

    else:
        # Retrieve form data and instantiate CustomData
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),  # Convert to float
                writing_score=float(request.form.get('writing_score'))  # Convert to float
            )

            # Validation for scores to ensure they're within the acceptable range
            if not (0 <= data.reading_score <= 100) or not (0 <= data.writing_score <= 100):
                flash("Scores must be between 0 and 100.", "error")
                return render_template('home.html')

            # Validation for non-empty string fields
            if not data.gender or not data.race_ethnicity or not data.parental_level_of_education or not data.lunch or not data.test_preparation_course:
                flash("All fields must be filled out.", "error")
                return render_template('home.html')

        except ValueError:
            flash("Invalid input. Please make sure all fields are filled out correctly.", "error")
            return render_template('home.html')

        # Prepare data for prediction
        pred_df = data.get_data_as_dataframe()
        logging.debug(f"Data for prediction: {pred_df}")

        # Call prediction pipeline and handle potential errors
        try:
            predict_pipeline = predictpipeline()
            results = predict_pipeline.predict(pred_df)
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            flash(f"Prediction failed: {e}", "error")
            return render_template('home.html')

        # Return the result to the home page
        return render_template('home.html', results=results[0])

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)



















