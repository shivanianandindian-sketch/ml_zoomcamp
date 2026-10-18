
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# Load the pinned 2026 dataset
url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv"
df = pd.read_csv(url)

# Separate categorical and numerical columns
target = "converted"
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
numerical_features = [col for col in numerical_cols if col != target]

# Fill missing values as instructed
df[categorical_cols] = df[categorical_cols].fillna("NA")
df[numerical_features] = df[numerical_features].fillna(0.0)

# ---------------- Question 1 ----------------
print("Q1: industry mode =", df["industry"].mode()[0])

# ---------------- Question 2 ----------------
corr = df[numerical_features].corr()

pairs = [
    ("interaction_count", "lead_score"),
    ("number_of_courses_viewed", "lead_score"),
    ("number_of_courses_viewed", "interaction_count"),
    ("annual_income", "interaction_count"),
]

for a, b in pairs:
    print(f"Q2: {a} and {b} =", round(corr.loc[a, b], 4))

# ---------------- Split the data ----------------
df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42
)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=42
)

y_train = df_train[target]
y_val = df_val[target]

# The target must not be part of the features
X_train = df_train.drop(columns=[target]).copy()
X_val = df_val.drop(columns=[target]).copy()

# ---------------- Question 3 ----------------
mi_cols = X_train.select_dtypes(include=["object"]).columns

mi_scores = {}
for col in mi_cols:
    score = mutual_info_score(y_train, X_train[col])
    mi_scores[col] = round(score, 2)

print("Q3: mutual information scores =", mi_scores)
print("Q3: highest MI =", max(mi_scores, key=mi_scores.get))

# ---------------- Model helper ----------------
def prepare_data(X_train, X_val):
    # One-hot encode categorical features
    train_dicts = X_train.to_dict(orient="records")
    val_dicts = X_val.to_dict(orient="records")

    dv = DictVectorizer(sparse=True)
    train_encoded = dv.fit_transform(train_dicts).astype("float64")
    val_encoded = dv.transform(val_dicts).astype("float64")

    # Fix sparse index types for liblinear compatibility
    train_encoded.indices = train_encoded.indices.astype(np.int32)
    train_encoded.indptr = train_encoded.indptr.astype(np.int32)

    val_encoded.indices = val_encoded.indices.astype(np.int32)
    val_encoded.indptr = val_encoded.indptr.astype(np.int32)

    return train_encoded, val_encoded


def train_and_evaluate(X_train, X_val, C=1.0):
    train_encoded, val_encoded = prepare_data(X_train, X_val)

    model = LogisticRegression(
        solver="liblinear",
        C=C,
        max_iter=1000,
        random_state=42
    )
    model.fit(train_encoded, y_train)

    predictions = model.predict(val_encoded)
    return accuracy_score(y_val, predictions)


# ---------------- Question 4 ----------------
original_accuracy = train_and_evaluate(X_train, X_val)

print("Q4: validation accuracy =", round(original_accuracy, 2))
print("Q4: unrounded accuracy =", original_accuracy)

# ---------------- Question 5 ----------------
for feature in [
    "lead_source",
    "number_of_courses_viewed",
    "interaction_count"
]:
    X_train_reduced = X_train.drop(columns=[feature])
    X_val_reduced = X_val.drop(columns=[feature])

    reduced_accuracy = train_and_evaluate(
        X_train_reduced, X_val_reduced
    )

    difference = original_accuracy - reduced_accuracy
    print(
        f"Q5: remove {feature}: "
        f"accuracy={reduced_accuracy:.6f}, "
        f"difference={difference:.6f}"
    )

# ---------------- Question 6 ----------------
for C in [0.000001, 0.00001, 0.0001, 0.001]:
    accuracy = train_and_evaluate(X_train, X_val, C=C)
    print(f"Q6: C={C}, accuracy={accuracy:.3f}")
