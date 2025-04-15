from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('diabetes_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        gender = 1 if request.form['gender'] == 'male' else 0
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        bmi = weight / ((height / 100) ** 2)
        skin_thickness = float(request.form['skin_thickness'])
        change_in_weight = float(request.form['change_in_weight'])
        family_history = 1 if request.form['family_history'] == 'yes' else 0

        input_data = np.array([[age, gender, height, weight, bmi,
                               skin_thickness, change_in_weight, family_history]])

        prediction = model.predict(input_data)[0]
        output = "at risk of diabetes" if prediction == 1 else "not at risk of diabetes"
        return render_template('result.html', prediction_text=f'The patient is {output}.')
    except Exception as e:
        return render_template('result.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
