from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("diabetes_model.pkl", "rb"))

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def get_weight_trend(change):
    if change > 2:
        return "Significant Gain"
    elif change < -2:
        return "Significant Loss"
    else:
        return "Stable"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form["age"])
    gender = int(request.form["gender"])
    height = int(request.form["height"])
    weight = int(request.form["weight"])
    bmi = float(request.form["bmi"])
    skin_thickness = int(request.form["skin_thickness"])
    change_in_weight = int(request.form["change_in_weight"])
    family_history = int(request.form["family_history"])

    input_data = np.array([[age, gender, height, weight, bmi,
                            skin_thickness, change_in_weight, family_history]])
    
    prediction_prob = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    report = {
        "prediction": "Likely Diabetic" if prediction == 1 else "Not Likely Diabetic",
        "risk_score": round(prediction_prob * 100, 2),
        "bmi_category": get_bmi_category(bmi),
        "weight_trend": get_weight_trend(change_in_weight),
        "family_risk": "High" if family_history == 1 else "Low",
        "recommendation": "Please consult a healthcare provider for further tests." if prediction == 1 else "Maintain a healthy lifestyle and regular checkups."
    }

    return render_template("result.html", report=report)

if __name__ == "__main__":
    app.run(debug=True)
