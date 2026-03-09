# ensemble_model.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

def train_ensemble_model(X_train, y_train):
    """
    Trains an ensemble model combining Random Forest and Gradient Boosting.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    return ensemble

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the ensemble model and returns accuracy in percentage.
    """
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred) * 100
