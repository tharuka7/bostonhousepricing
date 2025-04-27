import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

### Load model
regmodel = pickle.load(open("regmodel.pkl", 'rb'))
scaler = pickle.load(open("scaling.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Get data from the request
        data = request.json['data']
        print(f"Received data: {data}")
        
        # Convert data values into a numpy array and reshape to (1, -1)
        new_data = np.array(list(data.values())).reshape(1, -1)
        print(f"Reshaped data: {new_data}")
        
        # Scale the data
        new_data_scaled = scaler.transform(new_data)
        
        # Predict using the regression model
        output = regmodel.predict(new_data_scaled)
        print(f"Prediction result: {output[0]}")
        
        # Return the result as JSON response
        return jsonify(output[0])
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
