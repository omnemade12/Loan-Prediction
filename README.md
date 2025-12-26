🏦 Home Loan Approval Prediction System

An end-to-end Machine Learning project that predicts whether a home loan application is likely to be Approved or Rejected based on applicant details such as income, credit history, education, employment status, and property area.

The project uses a Logistic Regression model wrapped inside a Scikit-learn Pipeline and is deployed using Streamlit for real-time predictions.

🚀 Project Highlights

✅ End-to-end ML Pipeline (Preprocessing + Model)

✅ Handles categorical & numerical features automatically

✅ Uses probability-based decision making

✅ Interactive Streamlit web application

✅ Clean, production-style project structure

✅ Suitable for portfolio, interviews, and deployment

🧠 Machine Learning Approach
🔹 Model Used

Logistic Regression

class_weight="balanced" to handle class imbalance

Why Logistic Regression?

Stable and interpretable probabilities

Widely used in banking & credit risk

Performs well on small and medium tabular datasets

🔹 Preprocessing (Handled Inside Pipeline)

Missing value imputation

Feature scaling using StandardScaler

Categorical encoding using OneHotEncoder

No manual feature engineering required during prediction

✔ All preprocessing is bundled inside a single Pipeline, preventing feature mismatch and data leakage.

📊 Dataset Information

Dataset: Home Loan Application Data

Target variable: Loan_Status

1 → Approved

0 → Rejected

Important Features

Credit_History

ApplicantIncome

CoapplicantIncome

LoanAmount (in thousands)

Education

Self_Employed

Property_Area

🗂️ Project Structure
ML-LOAN_DATA_PROJECT/
│
├── app.py                     # Streamlit web application
├── home_loan.py               # Model training script
├── Loan_data.csv              # Dataset
├── requirements.txt           # Project dependencies
├── outputs/
│   └── loan_pipeline.pkl      # Saved ML pipeline
└── README.md

⚙️ How to Run the Project Locally
1️⃣ Clone the Repository
git clone https://github.com/your-username/home-loan-approval.git
cd home-loan-approval

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model

This script trains the ML pipeline and saves it as a .pkl file.

python home_loan.py


Expected output:

ROC-AUC: <score>
Pipeline model saved successfully

4️⃣ Run the Streamlit App
streamlit run app.py


Open in browser:

http://localhost:8501

🧪 Sample Input (High Approval Probability)

Use the following values to test the model:

Feature	Value
Gender	Male
Married	Yes
Dependents	0
Education	Graduate
Self Employed	No
Applicant Income	15000
Coapplicant Income	5000
Loan Amount (in thousands)	75
Loan Amount Term	360
Credit History	1.0
Property Area	Semiurban
📈 Output Interpretation

The model outputs a probability score

Decision logic:

Probability ≥ 0.50 → ✅ Loan Approved

Probability < 0.50 → ❌ Loan Rejected

This approach reflects real-world risk-based decision systems, not rigid rule-based logic.

🧩 Why Scikit-learn Pipeline?

Using a Pipeline ensures:

No feature mismatch between training and inference

No manual encoding errors

Single deployable .pkl file

Clean and production-ready ML workflow

🧑‍💻 Technologies Used

Python

Pandas

NumPy

Scikit-learn

Streamlit

Joblib

📌 Future Enhancements

Feature importance visualization

Probability confidence bands (Low / Medium / High risk)

Threshold tuning based on business rules

Model comparison (Random Forest, Gradient Boosting)

Deployment on Streamlit Cloud

🎯 Interview Talking Points

Why Logistic Regression is preferred in finance

Importance of probability-based predictions

How Pipelines prevent data leakage

Handling categorical and numerical features correctly

Difference between model accuracy and business decision thresholds

👤 Author

Om Nemade
Aspiring Data Scientist | ML & GenAI Enthusiast

⭐ If you find this project useful

Give it a ⭐ on GitHub — it helps a lot!
