Loan Approval Prediction Web App
================================

Project Overview
----------------
This project predicts loan approval for applicants using a machine learning model trained on the Kaggle Loan Prediction dataset (`train_u6lujuX_CVtuZ9i (1).csv`).  
The backend is built with Python (Flask) and uses an XGBoost model wrapped in a Scikit-learn pipeline.  
A simple web interface collects user input (income, loan amount, credit history, etc.) and returns an "Approved" or "Rejected" prediction with an approval probability.

Main Features
-------------
- Data preprocessing (handling missing values, encoding categorical variables).
- Model training using XGBoost classifier inside a Scikit-learn Pipeline.
- Evaluation with metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
- Web form for entering applicant details.
- Real-time prediction of loan approval with probability.
- Deployed as a live demo using Render (Gunicorn + Flask).

Tech Stack
----------
- Python 3
- Pandas, Scikit-learn, XGBoost, Joblib
- Flask (web framework)
- HTML/CSS (frontend template)
- Gunicorn (for production deployment)
- Render (cloud hosting)

Project Structure
-----------------
- train_model.py        : Script to load data, preprocess, train the model, and save `loan_model.pkl`.
- train_u6lujuX_CVtuZ9i (1).csv : Training dataset (Kaggle Loan Prediction).
- app.py                : Flask application that loads the trained model and serves the web UI.
- template/index.html   : HTML template for the loan prediction form and result display.
- requirements.txt      : Python dependencies for the project.

How the Model Works
-------------------
1. Loads the Kaggle loan dataset.
2. Drops the `Loan_ID` column and converts `Loan_Status` from Y/N to 1/0.
3. Selects important features:
   - Categorical: Gender, Married, Dependents, Education, Self_Employed, Property_Area
   - Numerical: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History
4. Fills missing values in key numeric columns (LoanAmount, Loan_Amount_Term, Credit_History).
5. Uses a ColumnTransformer:
   - OneHotEncoder for categorical features.
   - Pass-through for numerical features.
6. Trains an XGBoostClassifier within a Pipeline.
7. Evaluates the model with a train/test split and prints classification report + ROC-AUC.
8. Saves the trained pipeline and feature list as `loan_model.pkl` using Joblib.

Local Installation and Setup
----------------------------
1. Clone or download this repository.

2. Open a terminal / PowerShell and go to the project folder:
   cd "d:\Loan Prediction"
   (or the path where the project is located)

3. (Optional but recommended) Create and activate a virtual environment:
   python -m venv venv
   venv\Scripts\activate

4. Install dependencies:
   pip install -r requirements.txt

Training the Model
------------------
1. Make sure the dataset file `train_u6lujuX_CVtuZ9i (1).csv` is in the project root.

2. Run the training script:
   python train_model.py

3. The script will:
   - Load and preprocess the data.
   - Train the XGBoost model.
   - Print evaluation metrics.
   - Create `loan_model.pkl` in the project folder.

Running the Web Application (Locally)
-------------------------------------
1. Ensure `loan_model.pkl` exists (run train_model.py first if needed).

2. Start the Flask app:
   python app.py

3. Open your browser and go to:
   http://127.0.0.1:5000

4. Fill in the form fields:
   - Gender, Married, Dependents, Education, Self_Employed
   - ApplicantIncome, CoapplicantIncome
   - LoanAmount, Loan_Amount_Term
   - Credit_History (0 or 1)
   - Property_Area

5. Submit the form to see:
   - Prediction: Approved / Rejected
   - Approval probability in percentage.

Deployment (Render) - Summary
-----------------------------
1. Ensure `gunicorn` is in `requirements.txt` (already added).
2. Push the project to GitHub.
3. On Render:
   - Create a new Web Service from the GitHub repo.
   - Build command:  pip install -r requirements.txt
   - Start command:  gunicorn app:app
4. After deployment, Render provides a public URL (for example):
   https://loan-prediction-p2zb.onrender.com
   Use this as the demo link to share your project.

Usage Note
----------
- First request on the Render demo may take some time because the free instance needs to "wake up".
- Locally, the app responds immediately as long as `python app.py` is running.

- Deployed link
- https://loan-prediction-p2zb.onrender.com

Author
------
- Name: Anshu Gudadhe
- GitHub: https://github.com/anshugudadhe
