# data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    """
    Loads and preprocesses the dataset.
    """
    df = pd.read_csv(file_path)
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Split features and target
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
