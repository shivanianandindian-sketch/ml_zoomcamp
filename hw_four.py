
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score
)

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

# Load the 2026 lead-scoring dataset
df = pd.read_csv("course_lead_scoring_2026.csv")

# Check missing values
print("Missing values before filling:")
print(df.isnull().sum())

# Fill missing values according to feature type
for col in df.columns:
    if df[col].dtype == "object" or isinstance(df[col].dtype, pd.StringDtype):
        df[col] = df[col].fillna("NA")
    else:
        df[col] = df[col].fillna(0.0)

# Split the data exactly as specified
df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=1
)

df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1
)

# Question 1: ROC AUC feature importance
numerical_features = [
    "lead_score",
    "number_of_courses_viewed",
    "interaction_count",
    "annual_income"
]

y_train = df_train["converted"]

for col in numerical_features:
    auc = roc_auc_score(y_train, df_train[col])
    
    # Reverse the direction if AUC is below 0.5
    if auc < 0.5:
        auc = roc_auc_score(y_train, -df_train[col])

    print(f"{col}: AUC = {auc:.3f}")


# Prepare features and target
categorical = [
    "lead_source", "industry", "employment_status", "location"
]
numerical = [
    "annual_income", "number_of_courses_viewed",
    "interaction_count", "lead_score"
]
features = categorical + numerical

y_train = df_train["converted"].values
y_val = df_val["converted"].values

# One-hot encoding using DictVectorizer
dv = DictVectorizer(sparse=True)

train_dicts = df_train[features].to_dict(orient="records")
val_dicts = df_val[features].to_dict(orient="records")

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

# Fix sparse matrix index dtype for compatibility
X_train.indices = X_train.indices.astype(np.int32)
X_train.indptr = X_train.indptr.astype(np.int32)

X_val.indices = X_val.indices.astype(np.int32)
X_val.indptr = X_val.indptr.astype(np.int32)

# Train logistic regression
model = LogisticRegression(
    solver="liblinear",
    C=1.0,
    max_iter=1000
)
model.fit(X_train, y_train)

# Question 2: Validation ROC AUC
y_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)

print()
print(f"Q2: validation AUC = {auc:.3f}")


# Question 3: Precision and Recall
thresholds = np.arange(0.0, 1.01, 0.01)

best_threshold = None
smallest_difference = float("inf")

for threshold in thresholds:
    y_pred_binary = (y_pred >= threshold).astype(int)

    precision = precision_score(
        y_val, y_pred_binary, zero_division=0
    )
    recall = recall_score(
        y_val, y_pred_binary, zero_division=0
    )

    # Ignore thresholds where both metrics are zero
    if precision == 0 and recall == 0:
        continue

    difference = abs(precision - recall)

    if difference < smallest_difference:
        smallest_difference = difference
        best_threshold = threshold
        best_precision = precision
        best_recall = recall

print()
print(
    f"Q3: intersection threshold = {best_threshold:.2f}, "
    f"precision = {best_precision:.3f}, "
    f"recall = {best_recall:.3f}"
)


# Question 4: Find the threshold with maximum F1 score
best_f1 = -1
best_f1_threshold = None

for threshold in np.arange(0.0, 1.01, 0.01):
    y_pred_binary = (y_pred >= threshold).astype(int)

    score = f1_score(y_val, y_pred_binary, zero_division=0)

    if score > best_f1:
        best_f1 = score
        best_f1_threshold = threshold

print()
print(
    f"Q4: best F1 threshold = {best_f1_threshold:.2f}, "
    f"F1 score = {best_f1:.3f}"
)


#Five-fold cross-validation
def evaluate_c(logreg_c):
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []

    for train_idx, val_idx in kf.split(df_full_train):
        # Split the full training data into fold train and validation
        fold_train = df_full_train.iloc[train_idx]
        fold_val = df_full_train.iloc[val_idx]

        y_fold_train = fold_train["converted"].values
        y_fold_val = fold_val["converted"].values

        # Fit a fresh DictVectorizer on each fold's training data
        fold_dv = DictVectorizer(sparse=True)

        X_fold_train = fold_dv.fit_transform(
            fold_train[features].to_dict(orient="records")
        )
        X_fold_val = fold_dv.transform(
            fold_val[features].to_dict(orient="records")
        )

        # Fix sparse matrix index dtype for compatibility
        for X in (X_fold_train, X_fold_val):
            X.indices = X.indices.astype(np.int32)
            X.indptr = X.indptr.astype(np.int32)

        fold_model = LogisticRegression(
            solver="liblinear",
            C=logreg_c,
            max_iter=1000
        )
        fold_model.fit(X_fold_train, y_fold_train)

        fold_pred = fold_model.predict_proba(X_fold_val)[:, 1]
        scores.append(roc_auc_score(y_fold_val, fold_pred))

    return np.array(scores)


# Question 5: Standard deviation across five folds
scores_q5 = evaluate_c(1.0)

print()
print(f"Q5: fold AUC scores = {np.round(scores_q5, 3)}")
print(f"Q5: mean AUC = {scores_q5.mean():.3f}")
print(f"Q5: standard deviation = {scores_q5.std():.3f}")


# Question 6: Compare different C values
print()
print("Q6: Hyperparameter tuning")

for c in [0.000001, 0.001, 1]:
    scores = evaluate_c(c)

    print(
        f"C={c}: mean AUC = {scores.mean():.3f}, "
        f"std = {scores.std():.3f}"
    )


