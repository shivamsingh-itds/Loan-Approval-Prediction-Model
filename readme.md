# Loan Approval Prediction Model 🏦

## 📌 Project Overview
This project focuses on building an end-to-end machine learning pipeline to predict loan approval outcomes based on applicant demographic, financial, employment, and credit-related information. The objective is to assist financial institutions in identifying high-risk loan applicants while minimizing potential defaults.

The project follows a structured ML workflow including data ingestion, exploratory data analysis (EDA), feature engineering, model training, evaluation, and model persistence for deployment readiness.

---

## 🎯 Problem Statement
1. Predict whether a loan application is likely to be approved or rejected.
2. Identify key factors that influence loan approval decisions and credit risk.

---

## 📊 Dataset Description
The dataset contains approximately **45,000 loan application records** with a mix of numerical and categorical features. It represents real-world lending scenarios and customer profiles.

### Key Features
- **Demographic Information**:  
  `person_age`, `person_gender`, `person_education`

- **Financial Information**:  
  `person_income`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`

- **Employment & Stability**:  
  `person_emp_exp`, `person_home_ownership`

- **Credit History**:  
  `credit_score`, `cb_person_cred_hist_length`, `previous_loan_defaults_on_file`

- **Loan Purpose**:  
  `loan_intent`

- **Target Variable**:  
  `loan_status` (Binary classification)

The dataset contains **no missing values**, exhibits **non-normal distributions**, and shows **moderate class imbalance (~22%)**, which reflects realistic financial data behavior.

---

## 🔍 Exploratory Data Analysis (EDA)
Key EDA insights include:
- Credit score is the strongest indicator of loan approval.
- Higher loan-to-income ratios are associated with higher rejection rates.
- Applicants with previous loan defaults show significantly higher risk.
- Income and loan amount distributions are right-skewed, requiring robust scaling.

---

## 🛠 Feature Engineering
The following features were engineered to improve model performance:
- **Log Income**: Handles skewness in income distribution.
- **Loan-to-Income Ratio**: Measures repayment burden.
- **Employment Experience Ratio**: Indicates financial stability.
- **Credit History Score**: Combines credit score and history length.

---

## 🤖 Model Building
Multiple classification models were evaluated:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- K-Nearest Neighbors

### Final Model Selection
**Logistic Regression** was selected as the final model due to:
- Superior recall for high-risk applicants
- Lower false negatives (critical in loan risk assessment)
- Better interpretability and regulatory suitability

Class imbalance was handled using **class weighting**, without applying SMOTE.

---

## 📈 Model Evaluation
Evaluation metrics used:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Threshold tuning was considered to further reduce false negatives and improve risk sensitivity.

---

## 💾 Model Saving
The trained pipeline was saved using `joblib`:
This allows easy reuse and deployment.

---
## ✅ Conclusion
This project demonstrates how machine learning can be effectively applied to real-world loan approval and credit risk assessment problems. By prioritizing recall and minimizing false negatives, the model aligns with industry practices where the cost of approving risky loans outweighs rejecting safe applicants.

---
## 🚀 How to Run the Project

1. Clone the repository

```
git clone https://github.com/shivamsingh-itds/Loan-Approval-Prediction-Model.git
cd Loan-Approval-Prediction-Model
```
2.  Create & activate virtual environment
```
python -m venv ML
ML\Scripts\activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Train the model
```
python -m src.model_pipeline
```

---

## 🚀 Future Improvements
- Hyperparameter tuning
- Advanced threshold optimization
- Model explainability using SHAP
- Deployment using FastAPI or Streamlit

---

## 👤 Author

**Shivam Singh**
Aspiring Data Scientist | Machine Learning Enthusiast

🔗 GitHub: [https://github.com/shivamsingh-itds]
🔗 LinkedIn: [www.linkedin.com/in/shivamsinghds]

---

⭐ If you find this project helpful, feel free to star the repository!
