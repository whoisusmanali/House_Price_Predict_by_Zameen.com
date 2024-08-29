#/Users/usmanali/Uwindsor/AI/House_Price_Predict_by_Zameen.com/Python files/House_Price_Prediction_with_KNN.pkl

from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the KNN model
with open('/Users/usmanali/Uwindsor/AI/House_Price_Predict_by_Zameen.com/Flask_Server/Python files/House_Price_Prediction_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define unique values for one-hot encoding
unique_values = {
    'property_type': ['Flat', 'House', 'Penthouse', 'Lower Portion', 'Upper Portion', 'Room', 'Farm House'],
    'city': ['Islamabad', 'Lahore', 'Faisalabad', 'Rawalpindi', 'Karachi'],
    'purpose': ['For Rent', 'For Sale']
}

# Define the feature names
feature_names = ['property_type', 'city', 'baths', 'purpose', 'bedrooms', 'Area_in_Marla']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    df = pd.DataFrame([data])

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=['property_type', 'city', 'purpose'])

    # Add missing columns if any
    for feature, values in unique_values.items():
        for value in values:
            col_name = f"{feature}_{value}"
            if col_name not in df_encoded.columns:
                df_encoded[col_name] = 0

    # Ensure the DataFrame has the correct order of columns
    expected_columns = [f"{feature}_{value}" for feature, values in unique_values.items() for value in values]
    expected_columns += ['baths', 'bedrooms', 'Area_in_Marla']
    df_encoded = df_encoded.reindex(columns=expected_columns, fill_value=0)

    # Predict
    prediction = model.predict(df_encoded)

    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
