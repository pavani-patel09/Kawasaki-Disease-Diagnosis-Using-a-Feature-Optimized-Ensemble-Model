# main.py

from data_preprocessing import preprocess_data
from ensemble_model import train_ensemble_model, evaluate_model

def main():
    """
    Main function to run the Kawasaki Disease Diagnosis pipeline.
    Steps:
    1. Load and preprocess dataset
    2. Train ensemble model
    3. Evaluate model accuracy
    """

    print("Kawasaki Disease Diagnosis Running...")

    # Load and preprocess the dataset
    X_train, X_test, y_train, y_test = preprocess_data('datasets/kawasaki_data.csv')
    
    # Train the ensemble model
    model = train_ensemble_model(X_train, y_train)
    
    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Ensemble model accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
