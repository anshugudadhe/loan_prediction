from flask import Flask, render_template, request
import joblib
import pandas as pd

# Use existing "template" folder for HTML files
app = Flask(__name__, template_folder="template")

# Load trained model (make sure you ran train_model.py first)
bundle = joblib.load("loan_model.pkl")
model = bundle["model"]
feature_cols = bundle["feature_cols"]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        form_data = {
            "Gender": request.form.get("Gender"),
            "Married": request.form.get("Married"),
            "Dependents": request.form.get("Dependents"),
            "Education": request.form.get("Education"),
            "Self_Employed": request.form.get("Self_Employed"),
            "ApplicantIncome": float(request.form.get("ApplicantIncome") or 0),
            "CoapplicantIncome": float(request.form.get("CoapplicantIncome") or 0),
            "LoanAmount": float(request.form.get("LoanAmount") or 0),
            "Loan_Amount_Term": float(request.form.get("Loan_Amount_Term") or 360),
            "Credit_History": float(request.form.get("Credit_History") or 1),
            "Property_Area": request.form.get("Property_Area"),
        }

        X_new = pd.DataFrame([form_data], columns=feature_cols)
        proba = model.predict_proba(X_new)[0, 1]
        pred = int(proba >= 0.5)

        prediction = "Approved" if pred == 1 else "Rejected"
        probability = round(float(proba) * 100, 2)

    return render_template("index.html", prediction=prediction, probability=probability)


if __name__ == "__main__":
    app.run(debug=True)

