import pandas as pd
import os
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# LOAD DATA
df = pd.read_csv("Loan_data.csv")
df.drop(columns=["Loan_ID"], inplace=True)

# HANDLE MISSING
for col in df.select_dtypes(include="object"):
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in df.select_dtypes(exclude="object"):
    df[col].fillna(df[col].median(), inplace=True)

df["Dependents"] = df["Dependents"].replace("3+", "3").astype(int)
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

# PIPELINE
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ]
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ))
])

# TRAIN
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# EVALUATE
probs = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, probs))

# SAVE SINGLE ARTIFACT
os.makedirs("outputs", exist_ok=True)
dump(model, "outputs/loan_pipeline.pkl")

print("Pipeline model saved successfully")
