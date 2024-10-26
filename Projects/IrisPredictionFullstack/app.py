import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

# Create flask app
app = Flask(__name__)

# Load your model
model = pickle.load(open('D:\IMPORTANT\AIML\Projects\IrisPredictionFullstack\model.pkl', 'rb'))


# Route the home page
@app.route('/')
def home():
    return render_template('index.html')


# Route the predict url
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    iFeatures = [float(data) for data in request.form.values()]
    iFeatures = [np.array(iFeatures)]

    prediction = model.predict(iFeatures)

    return render_template('index.html', prediction_text=f"The Flower most likely belongs to {prediction[0]} Family")


# Run the application
if (__name__ == '__main__'):
    app.run(debug='True')
