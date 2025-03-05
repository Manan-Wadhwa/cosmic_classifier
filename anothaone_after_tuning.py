import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib  # For saving and loading models

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

def load_and_preprocess_data(train_path):
    print(f"Loading data from '{train_path}'...")
    data = pd.read_csv(train_path)
    print(f"Data loaded with shape: {data.shape}")
    
    print("Replacing sentinel values (-999) with NaN...")
    data.replace({-999: np.nan}, inplace=True)
    
    print("Converting categorical columns if needed...")
    category_columns = ['Magnetic Field Strength', 'Radiation Levels']
    for col in category_columns:
        if col in data and data[col].dtype == 'object':
            data[col] = data[col].str.replace('Category_', '').astype(float)
    
    print("Dropping rows with missing target variable 'Prediction'...")
    data.dropna(subset=['Prediction'], inplace=True)
    
    print("Separating features and target...")
    X = data.drop(columns=['Prediction'])
    y = data['Prediction']
    
    print("Imputing missing values in features using median strategy...")
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    print(f"Imputation complete. Final feature shape: {X.shape}")
    
    return X, y

def feature_diagnostics(X, y):
    print("Generating class distribution plot...")
    plt.figure(figsize=(8, 5))
    y.value_counts(normalize=True).plot(kind='bar', title='Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Proportion')
    plt.savefig('class_distribution.png')
    plt.close()
    print("Class distribution plot saved as 'class_distribution.png'.")
    
    print("Generating correlation heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()
    print("Correlation heatmap saved as 'correlation_heatmap.png'.")
    
    print("Calculating feature importance using RandomForestClassifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importance = pd.DataFrame({
        'Feature': X.columns, 
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importances')
    plt.savefig('feature_importances.png')
    plt.close()
    print("Feature importance plot saved as 'feature_importances.png'.")

def main():
    print("=== Process Started: Data Loading and Preprocessing ===")
    X, y = load_and_preprocess_data("cosmicclassifierTraining.csv")
    print("=== Data Loading and Preprocessing Completed ===\n")
    
    print("=== Running Feature Diagnostics ===")
    feature_diagnostics(X, y)
    print("=== Feature Diagnostics Completed ===\n")
    
    print("=== Splitting Data into Training and Validation Sets ===")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set shape: {X_train.shape} | Validation set shape: {X_val.shape}\n")
    
    print("=== Applying SMOTE to Balance the Training Data ===")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied. Resampled training set shape: {X_train_resampled.shape}\n")
    
    print("=== Defining Base Models for Stacking Classifier ===")
    base_models = [
        ('rf', RandomForestClassifier(random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ]
    
    print("=== Creating Stacking Classifier ===")
    stacking_classifier = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(multi_class='ovr', max_iter=1000),
        cv=5
    )
    
    print("=== Scaling Data ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled = scaler.transform(X_val)
    print("Data scaling complete.\n")
    
    # --- Hyperparameter Tuning with GridSearchCV ---
    print("=== Tuning Stacking Classifier with GridSearchCV ===")
    param_grid = {
        # Final estimator (Logistic Regression) hyperparameters
        'final_estimator__C': [0.1, 1, 10],
        'final_estimator__max_iter': [500, 1000],
        # Base model hyperparameters (accessed via their tuple names)
        'rf__n_estimators': [50, 100],
        'gb__n_estimators': [50, 100],
        'svm__C': [0.1, 1, 10]
    }
    grid_search = GridSearchCV(
        estimator=stacking_classifier,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='accuracy'
    )
    grid_search.fit(X_train_scaled, y_train_resampled)
    print("Tuning complete.")
    print("Best parameters found:", grid_search.best_params_)
    print("Best cross-validation accuracy:", grid_search.best_score_)
    
    best_model = grid_search.best_estimator_
    
    print("=== Making Predictions on Validation Set ===")
    y_pred = best_model.predict(X_val_scaled)
    
    print("\n--- MODEL PERFORMANCE ---")
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print(f"Overall Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    
    # --- Save the Best Model and Scaler ---
    print("=== Saving the Best Model and Scaler for Future Predictions ===")
    joblib.dump(best_model, "stacking_classifier_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model saved as 'stacking_classifier_model.pkl' and scaler saved as 'scaler.pkl'.")

if __name__ == "__main__":
    main()
