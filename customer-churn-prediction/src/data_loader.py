"""
Data Loader and Preprocessor for Customer Churn Prediction
This module handles data loading, cleaning, and preprocessing for ML models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import mysql.connector
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

class ChurnDataLoader:
    def __init__(self, data_path=None, db_config=None):
        """
        Initialize the data loader
        
        Args:
            data_path (str): Path to CSV file
            db_config (dict): Database configuration for SQL connection
        """
        self.data_path = data_path
        self.db_config = db_config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data_from_csv(self):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading data from CSV: {e}")
            return None
    
    def load_data_from_sql(self, query=None):
        """Load data from SQL database"""
        if not self.db_config:
            print("No database configuration provided")
            return None
            
        try:
            # Create database connection
            engine = create_engine(
                f"mysql+mysqlconnector://{self.db_config['user']}:"
                f"{self.db_config['password']}@{self.db_config['host']}:"
                f"{self.db_config['port']}/{self.db_config['database']}"
            )
            
            # Default query if none provided
            if not query:
                query = "SELECT * FROM telco_customers_clean"
            
            df = pd.read_sql(query, engine)
            print(f"Data loaded from SQL: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            print(f"Error loading data from SQL: {e}")
            return None
    
    def clean_data(self, df):
        """Clean and prepare the data"""
        print("Starting data cleaning...")
        
        # Handle missing values in TotalCharges
        if 'TotalCharges' in df.columns:
            # Convert TotalCharges to numeric, replacing empty strings with NaN
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
            # Fill NaN values with 0 or median
            df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        # Convert SeniorCitizen to categorical
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
        # Standardize column names (remove spaces, convert to lowercase)
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        
        print(f"Data cleaning completed. Shape: {df.shape}")
        return df
    
    def encode_categorical_features(self, df, target_column='churn'):
        """Encode categorical features using Label Encoding"""
        print("Encoding categorical features...")
        
        # Identify categorical columns (excluding target)
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        if target_column in categorical_columns:
            categorical_columns.remove(target_column)
        
        # Apply label encoding
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Encode target variable
        if target_column in df.columns:
            le_target = LabelEncoder()
            df[target_column] = le_target.fit_transform(df[target_column])
            self.label_encoders[target_column] = le_target
        
        print(f"Encoded {len(categorical_columns)} categorical features")
        return df
    
    def create_features(self, df):
        """Create additional features for better prediction"""
        print("Creating additional features...")
        
        # Feature: Average monthly charges per tenure month
        if 'monthlycharges' in df.columns and 'tenure' in df.columns:
            df['avg_monthly_charges_per_tenure'] = df['monthlycharges'] / (df['tenure'] + 1)
        
        # Feature: Total charges per tenure ratio
        if 'totalcharges' in df.columns and 'tenure' in df.columns:
            df['charges_per_tenure'] = df['totalcharges'] / (df['tenure'] + 1)
        
        # Feature: Service usage score (count of additional services)
        service_columns = ['onlinesecurity', 'onlinebackup', 'deviceprotection', 
                          'techsupport', 'streamingtv', 'streamingmovies']
        
        available_services = [col for col in service_columns if col in df.columns]
        if available_services:
            # Assuming 'Yes' is encoded as 1 and 'No' as 0
            df['service_usage_score'] = df[available_services].sum(axis=1)
        
        print("Feature creation completed")
        return df
    
    def prepare_for_modeling(self, df, target_column='churn', test_size=0.2, random_state=42):
        """Prepare data for machine learning models"""
        print("Preparing data for modeling...")
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature column names
        self.feature_columns = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale numerical features
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        if len(numerical_columns) > 0:
            X_train_scaled[numerical_columns] = self.scaler.fit_transform(X_train[numerical_columns])
            X_test_scaled[numerical_columns] = self.scaler.transform(X_test[numerical_columns])
        
        print(f"Training set: {X_train_scaled.shape[0]} samples")
        print(f"Test set: {X_test_scaled.shape[0]} samples")
        print(f"Features: {len(self.feature_columns)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_data_summary(self, df):
        """Generate data summary statistics"""
        print("\n=== DATA SUMMARY ===")
        print(f"Dataset shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n=== MISSING VALUES ===")
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(missing_values[missing_values > 0])
        else:
            print("No missing values found")
        
        print("\n=== DATA TYPES ===")
        print(df.dtypes.value_counts())
        
        if 'churn' in df.columns:
            print("\n=== TARGET DISTRIBUTION ===")
            churn_dist = df['churn'].value_counts()
            print(churn_dist)
            print(f"Churn rate: {(churn_dist.get('Yes', churn_dist.get(1, 0)) / len(df)) * 100:.2f}%")
        
        return df.describe()

def main():
    """Example usage of the ChurnDataLoader"""
    # Initialize data loader
    loader = ChurnDataLoader(data_path='../data/telco_customer_churn.csv')
    
    # Load and process data
    df = loader.load_data_from_csv()
    if df is not None:
        # Clean data
        df_clean = loader.clean_data(df)
        
        # Get data summary
        summary = loader.get_data_summary(df_clean)
        
        # Encode categorical features
        df_encoded = loader.encode_categorical_features(df_clean)
        
        # Create additional features
        df_features = loader.create_features(df_encoded)
        
        # Prepare for modeling
        X_train, X_test, y_train, y_test = loader.prepare_for_modeling(df_features)
        
        print("\nData preparation completed successfully!")
        return X_train, X_test, y_train, y_test, loader
    
    return None

if __name__ == "__main__":
    main()