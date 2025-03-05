import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Comprehensive Data Validation
def validate_dataset(data, stage='Initial'):
    """
    Perform comprehensive data validation
    """
    print(f"\n--- DATA VALIDATION ({stage}) ---")
    
    # Check for empty dataset
    if data.empty:
        raise ValueError("Dataset is empty!")
    
    # Basic information
    print(f"Total rows: {len(data)}")
    print(f"Total columns: {len(data.columns)}")
    
    # Check for NaNs
    print("\nNaN Check:")
    nan_counts = data.isna().sum()
    print(nan_counts[nan_counts > 0])
    
    # Check unique values in Prediction
    print("\nUnique Prediction Values:")
    print(data['Prediction'].value_counts())
    
    # Check data types
    print("\nData Types:")
    print(data.dtypes)
    
    return data

# Data Loading and Preprocessing
def load_and_preprocess_data(train_path):
    # Load data
    data = pd.read_csv(train_path)
    
    # Initial validation
    validate_dataset(data, 'Initial')
    
    # Replace sentinel values
    data.replace({-999: np.nan}, inplace=True)
    
    # Handle category columns
    category_columns = ['Magnetic Field Strength', 'Radiation Levels']
    for col in category_columns:
        if data[col].dtype == 'object':
            data[col] = data[col].str.replace('Category_', '').astype(float)
    
    # Remove rows with NaN in target variable
    data.dropna(subset=['Prediction'], inplace=True)
    
    # Separate features and target
    X = data.drop(columns=['Prediction'])
    y = data['Prediction']
    
    # Create imputer for features
    imputer = SimpleImputer(strategy='median')
    
    # Impute missing values in features
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X), 
        columns=X.columns, 
        index=X.index
    )
    
    # Validate after imputation
    validate_dataset(pd.concat([X_imputed, y], axis=1), 'After Imputation')
    
    return X_imputed, y

# Feature Diagnostics
def feature_diagnostics(X, y):
    # Class Distribution
    print("\n--- CLASS DISTRIBUTION ---")
    class_dist = pd.Series(y).value_counts(normalize=True)
    print(class_dist)
    
    plt.figure(figsize=(10, 6))
    class_dist.plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Proportion')
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()
    
    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    # Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n--- FEATURE IMPORTANCES ---")
    print(feature_importance)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()

# Main Execution
def main():
    try:
        # Load and preprocess data
        X, y = load_and_preprocess_data("cosmicclassifierTraining.csv")
        
        # Run feature diagnostics
        feature_diagnostics(X, y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Create base models
        base_models = [
            ('rf', RandomForestClassifier(random_state=42)),
            ('gb', GradientBoostingClassifier(random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ]

        # Create stacking classifier
        stacking_classifier = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(multi_class='ovr', max_iter=1000),
            cv=5
        )

        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_val_scaled = scaler.transform(X_val)
        
        # Fit stacking classifier
        stacking_classifier.fit(X_train_scaled, y_train_resampled)
        
        # Predict and evaluate
        y_pred = stacking_classifier.predict(X_val_scaled)
        
        print("\n--- MODEL PERFORMANCE ---")
        print("Classification Report:")
        print(classification_report(y_val, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        
        print(f"\nOverall Accuracy: {accuracy_score(y_val, y_pred)}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

# Run the main function
if __name__ == "__main__":
    main()