import joblib

def predict_result(df):
    # Load the trained model
    with open('kidney_disease_model.pkl', 'rb') as model_file:
        model = joblib.load(model_file)
    
    # Make a prediction
    prediction = model.predict(df)  # Assuming df is formatted for prediction (like a single row of features)
    
    # Convert prediction to a more readable format
    if prediction[0] == 1:
        return "Maladie rénale détectée"
    else:
        return "Pas de maladie rénale détectée"
