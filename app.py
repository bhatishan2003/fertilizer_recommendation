import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model
with open('Fertilizer_Recommendation_model_2.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the label encoder
label_encoder = LabelEncoder()
df = pd.read_csv("Fertilizer Prediction.csv")
df_encoded = pd.get_dummies(df, columns=['Soil Type', 'Crop Type'], drop_first=True)
label_encoder.fit(df_encoded['Fertilizer Name'])

# Map for specific fertilizers
fertilizer_map = {
    '28-28': 'Gromor 28-28',
    '14-35-14': 'Gromor 14-35-14',
    '17-17-17': 'Gromor 17-17-17',
    '20-20': 'Gromor 20-20',
    '10-26-26': 'Gromor 10-26-26',
    'Urea': 'Urea',
    'DAP': 'DAP'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = pd.DataFrame([data])
    input_data_encoded = pd.get_dummies(input_data, columns=['Soil Type', 'Crop Type'], drop_first=True)

    # Ensure all necessary columns are present
    for col in model.feature_names_in_:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0

    input_data_encoded = input_data_encoded[model.feature_names_in_]
    prediction = model.predict(input_data_encoded)
    predicted_fertilizer = label_encoder.inverse_transform(prediction)
    predicted_fertilizer_name = predicted_fertilizer[0]

    # Apply the fertilizer map
    predicted_fertilizer_name = fertilizer_map.get(predicted_fertilizer_name, predicted_fertilizer_name)

    return render_template('index.html', prediction_text=f'Predicted Fertilizer: {predicted_fertilizer_name}')

if __name__ == '__main__':
    app.run(debug=True)
