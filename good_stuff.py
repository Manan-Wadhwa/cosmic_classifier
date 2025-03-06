# Install necessary libraries (if needed)


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score
import optuna

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import torch

# -------------------------------
# 1. Data Loading & Preprocessing
# -------------------------------
df = pd.read_csv("cosmicclassifierTraining.csv")

# Separate target and features
target_col = "Prediction"
X = df.drop(columns=[target_col])
y = df[target_col]

# Remove rows with missing target
valid_rows = y.notna()
X, y = X[valid_rows], y[valid_rows]

# Identify categorical and numerical columns
categorical_cols = ["Magnetic Field Strength", "Radiation Levels"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Impute missing values
num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")
X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

# Encode categorical variables
encoder = LabelEncoder()
for col in categorical_cols:
    X[col] = encoder.fit_transform(X[col])

# Scale numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# -------------------------------
# 2. Train-Test Split & SMOTE Oversampling
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.2,
                                                  stratify=y,
                                                  random_state=42)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# -------------------------------
# 3. Hyperparameter Tuning with Optuna
# -------------------------------

# --- Tuning XGBoost ---
def objective_xgb(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'tree_method': 'gpu_hist',  # Use GPU acceleration
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    model = XGBClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=20)
best_params_xgb = study_xgb.best_params

xgboost = XGBClassifier(**best_params_xgb,
                        tree_method='gpu_hist',
                        use_label_encoder=False,
                        eval_metric='logloss')
xgboost.fit(X_train_resampled, y_train_resampled)

# --- Tuning CatBoost ---
def objective_cat(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'depth': trial.suggest_int('depth', 4, 10),
        'iterations': trial.suggest_int('iterations', 200, 1000),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10)
    }
    model = CatBoostClassifier(**params,
                               task_type="GPU",
                               loss_function='MultiClass',
                               verbose=0)
    model.fit(X_train_resampled, y_train_resampled,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=50,
              verbose=False)
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)

study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(objective_cat, n_trials=20)
best_params_cat = study_cat.best_params

catboost = CatBoostClassifier(**best_params_cat,
                              task_type="GPU",
                              loss_function='MultiClass',
                              verbose=0)
catboost.fit(X_train_resampled, y_train_resampled)

# --- Tuning LightGBM with try-except ---
def objective_lgb(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100)
    }
    try:
        # Try training with GPU
        model = LGBMClassifier(**params, device="gpu")
        model.fit(
            X_train_resampled,
            y_train_resampled,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
    except Exception as e:
        print("LightGBM GPU training failed with error:", e)
        # Fallback to CPU
        model = LGBMClassifier(**params, device="cpu")
        model.fit(
            X_train_resampled,
            y_train_resampled,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)

study_lgb = optuna.create_study(direction="maximize")
study_lgb.optimize(objective_lgb, n_trials=20)
best_params_lgb = study_lgb.best_params

try:
    lightgbm = LGBMClassifier(**best_params_lgb, device="gpu")
    lightgbm.fit(X_train_resampled, y_train_resampled)
except Exception as e:
    print("Final LightGBM GPU training failed with error:", e)
    lightgbm = LGBMClassifier(**best_params_lgb, device="cpu")
    lightgbm.fit(X_train_resampled, y_train_resampled)

# -------------------------------
# 4. Stacking Ensemble
# -------------------------------
stack_model = StackingClassifier(
    estimators=[('xgb', xgboost), ('lgbm', lightgbm), ('cat', catboost)],
    final_estimator=LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000),
    passthrough=True,
    n_jobs=1  # Use 1 to avoid parallel GPU conflicts
)
stack_model.fit(X_train_resampled, y_train_resampled)

# -------------------------------
# 5. Evaluation
# -------------------------------
y_pred = stack_model.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, stack_model.predict_proba(X_val), multi_class='ovr')

print(f"🚀 Validation Accuracy: {accuracy_val:.4f}")
print(f"🚀 Validation ROC AUC: {roc_auc:.4f}")

train_pred = stack_model.predict(X_train_resampled)
train_accuracy = accuracy_score(y_train_resampled, train_pred)
print(f"🚀 Training Accuracy: {train_accuracy:.4f}")
