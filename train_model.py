import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("train_u6lujuX_CVtuZ9i (1).csv")

if "Loan_ID" in df.columns:
    df = df.drop(columns=["Loan_ID"])

df = df.dropna(subset=["Loan_Status"])
df["Loan_Status"] = (df["Loan_Status"] == "Y").astype(int)

feature_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

X = df[feature_cols].copy()
y = df["Loan_Status"]

X["LoanAmount"] = X["LoanAmount"].fillna(X["LoanAmount"].median())
X["Loan_Amount_Term"] = X["Loan_Amount_Term"].fillna(X["Loan_Amount_Term"].median())
X["Credit_History"] = X["Credit_History"].fillna(X["Credit_History"].mode()[0])

cat_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]
num_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
)

clf = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

joblib.dump({"model": clf, "feature_cols": feature_cols}, "loan_model.pkl")
print("Saved model to loan_model.pkl")