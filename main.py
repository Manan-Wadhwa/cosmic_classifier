import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# Load data
train_data = pd.read_csv("cosmicclassifierTraining.csv")
test_data = pd.read_csv("cosmicclassifierTest.csv")
print("flag1")

print(train_data.head())

def preprocess_data(data):
    data.replace({-999: np.nan}, inplace=True)

    # Only drop rows with NaN values in 'Prediction' if 'Prediction' is a column
    if 'Prediction' in data.columns:
        data.dropna(subset=['Prediction'], inplace=True)

    # Handle category 9
    data['Magnetic Field Strength'] = data['Magnetic Field Strength'].apply(lambda x: x.replace('Category_', '') if isinstance(x, str) else x)
    data['Radiation Levels'] = data['Radiation Levels'].apply(lambda x: x.replace('Category_', '') if isinstance(x, str) else x)
    data['Magnetic Field Strength'] = data['Magnetic Field Strength'].astype(float)
    data['Radiation Levels'] = data['Radiation Levels'].astype(float)

    imputer = SimpleImputer(strategy='median')

    # If 'Prediction' is a column, exclude it from imputation
    features = data.drop(columns=['Prediction'] if 'Prediction' in data else [])

    features_imputed = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)
    data.update(features_imputed)
    return data

# Preprocess training and test data
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Separate features and target
X = train_data.drop(columns=['Prediction'])
y = train_data['Prediction']

# Before scaling, ensure there are no NaNs in X
X.fillna(X.mean(), inplace=True)  # Or use another imputation strategy if you prefer

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE for balancing classes
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define multiple models
models = [
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("Gradient Boosting", GradientBoostingClassifier()),
    ("AdaBoost", AdaBoostClassifier()),
    ("SVM", SVC()),
    ("KNN", KNeighborsClassifier(n_neighbors=10)),
    ("Decision Tree", DecisionTreeClassifier())
]

# Train and evaluate models
results = []
count = 0
for name, model in models:
    count+=1
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='macro')
    cm = confusion_matrix(y_val, y_pred)
    results.append((name, accuracy, precision, cm))
    print(f"Model {count} trained and evaluated")

# Sort results by precision
results.sort(key=lambda x: x[2], reverse=True)

# Display results
for result in results:
    print(f"Model: {result[0]}")
    print(f"Accuracy: {result[1]}")
    print(f"Precision: {result[2]}")
    print(f"Confusion Matrix:\n{result[3]}\n")

# Hyperparameter tuning for the best model
best_model_name, best_model = models[0]
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_resampled, y_train_resampled)

print(f"Best parameters for {best_model_name}: {grid_search.best_params_}")
print(f"Best accuracy: {grid_search.best_score_}")