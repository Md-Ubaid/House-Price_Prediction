from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load our saved model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    # Show the HTML page when someone visits the site
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get the area typed into the web form
    house_area = float(request.form['area'])
    
    # 2. Format it for the model (2D array)
    input_features = np.array([[house_area]])
    
    # 3. Make the prediction
    prediction = model.predict(input_features)[0]
    
    # 4. Send the result back to the HTML page
    return render_template('index.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)