import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Example: Load dataset (replace with your actual dataset)
df = pd.read_csv('./data/kidney_disease.csv')

# Example preprocessing (replace with your actual preprocessing steps)
X = df.drop('class', axis=1)  # Features
y = df['class']  # Target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model to a file
joblib.dump(model, 'kidney_disease_model.pkl')
