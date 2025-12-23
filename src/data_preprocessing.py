import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    # Target
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    # Feature Engineering

    X["log_income"] = np.log1p(X["person_income"])
    X["loan_income_ratio"] = X["loan_amnt"] / X["person_income"]
    X["emp_exp_ratio"] = X["person_emp_exp"] / X["person_age"]
    X["credit_history_score"] = (
        X["credit_score"] * X["cb_person_cred_hist_length"]
    )
    # Column Types
    
    numerical_cols = [
        "person_age",
        "person_income",
        "person_emp_exp",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "credit_score",
        "log_income",
        "loan_income_ratio",
        "emp_exp_ratio",
        "credit_history_score"
    ]

    categorical_cols = [
        "person_gender",
        "person_education",
        "person_home_ownership",
        "loan_intent",
        "previous_loan_defaults_on_file"
    ]

 
    # Transformers
   
    num_transformer = RobustScaler()

    cat_transformer = OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_cols),
            ("cat", cat_transformer, categorical_cols)
        ]
    )
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor
