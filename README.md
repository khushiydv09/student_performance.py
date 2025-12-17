# Student Performance Prediction using Machine Learning

## Overview
This project demonstrates the application of machine learning techniques to solve a real-world problem.

## Problem Statement
Brief description of the problem.

## Approach
- Data preprocessing
- Model training
- Evaluation

## Technologies Used
Python, scikit-learn, Pandas

## Results
The model produces meaningful predictions based on input features.

## Future Improvements
- Use larger real-world datasets
- Improve feature engineering
- Address bias and ethical considerations
  CODE-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'attendance': [90, 75, 85, 60, 95, 70, 80],
    'study_hours': [4, 2, 3, 1, 5, 2, 3],
    'previous_score': [85, 65, 78, 50, 92, 60, 75],
    'pass': [1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Features and target
X = df[['attendance', 'study_hours', 'previous_score']]
y = df['pass']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
