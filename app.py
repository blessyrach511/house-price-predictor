from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bhk = int(request.form['bhk'])
    location = int(request.form['location'])  # from encoded dropdown

    prediction = model.predict(np.array([[area, bhk, location]]))[0]
    return render_template('index.html', prediction_text=f'Estimated Price: ₹ {round(prediction, 2)} Lakhs')

if __name__ == '__main__':
    app.run(debug=True)
