# Day 3 - Credit Risk Analysis

# pip install pandas numpy matplotlib scikit-learn statsmodels plotly-express plotly

import pandas as pd
import numpy as np

# ------------------ Synthetic Data ------------------
np.random.seed(42)
credit_scores = np.random.randint(300, 850, size=1000)
loan_amounts = np.random.randint(1000, 50000, size=1000)

# Function to assign higher probability of default to lower credit scores
def assign_default_probability(credit_score):
    if credit_score < 500:
        return np.random.choice([0, 1], p=[0.3, 0.7])
    elif credit_score < 700:
        return np.random.choice([0, 1], p=[0.6, 0.4])
    else:
        return np.random.choice([0, 1], p=[0.8, 0.2])

default = np.array([assign_default_probability(score) for score in credit_scores])

data = pd.DataFrame({'CreditScore': credit_scores, 'LoanAmount': loan_amounts, 'Default': default})
print("Synthetic credit risk data created.")

# ------------------ Plot ------------------
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
colors = {0: 'blue', 1: 'red'}
plt.scatter(data['CreditScore'], data['LoanAmount'], c=data['Default'].apply(lambda x: colors[x]), alpha=0.5)
plt.xlabel('Credit Score')
plt.ylabel('Loan Amount')
plt.title('Credit Score vs Loan Amount by Default Status')
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='No Default'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Default')])

plt.show()

# ------------------ Credit Risk Analysis ------------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = data[['CreditScore', 'LoanAmount']]
y = data['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Credit risk analysis accuracy: {accuracy:.2f}')

# ------------------ Plot Predictions ------------------
# Scatter plot of actual vs predicted default status
plt.figure(figsize=(10, 6))
plt.scatter(X_test['CreditScore'], X_test['LoanAmount'], c=y_test.apply(lambda x: 'blue' if x == 0 else 'red'), alpha=0.5, label='Actual')
plt.scatter(X_test['CreditScore'], X_test['LoanAmount'], c=predictions, alpha=0.8, marker='x', label='Predicted')
plt.xlabel('Credit Score')
plt.ylabel('Loan Amount')
plt.title('Actual vs Predicted Default Status')
plt.legend()
plt.show()

# ------------------ Predictions ------------------
def predict_default(credit_score, loan_amount, model):
    input_data = pd.DataFrame({'CreditScore': [credit_score], 'LoanAmount': [loan_amount]})
    prediction = model.predict(input_data)[0]
    return prediction

# Example usage
credit_score = 600
loan_amount = 2000
prediction = predict_default(credit_score, loan_amount, model)
print(f'Predicted default status for a customer with credit score {credit_score} and loan amount {loan_amount}: {prediction}')

# ------------------ Recommendations ------------------
def recommend_loan_approval(credit_score, loan_amount, model):
    input_data = pd.DataFrame({'CreditScore': [credit_score], 'LoanAmount': [loan_amount]})
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return 'Recommend loan approval with caution.'
    else:
        return 'Recommend loan approval.'

# Example usage
credit_score = 600
loan_amount = 2000
recommendation = recommend_loan_approval(credit_score, loan_amount, model)
print(recommendation)

# ------------------ Model Evaluation ------------------

from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix
# ----------------
#                   | Predicted No Default | Predicted Default |
# ----------------------------------------------------------
# Actual No Default   | True Negative (TN) | False Positive (FP)|
# ----------------------------------------------------------
# Actual Default      | False Negative (FN)| True Positive (TP) |


# Confusion matrix
cm = confusion_matrix(y_test, predictions)
print('Confusion Matrix:')
print(cm)



# Classification report
cr = classification_report(y_test, predictions)
print('\nClassification Report:')

print(cr)
