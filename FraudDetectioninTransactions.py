
import pandas as pd
import numpy as np


np.random.seed(42)


amounts = np.random.normal(loc=500, scale=200, size=1000)
amounts = np.clip(amounts, 10, 1000)  

fraud_prob = np.where(amounts > 700, 0.9, 0.1)
fraud = np.random.binomial(1, fraud_prob)

data = pd.DataFrame({'TransactionAmount': amounts, 'Fraud': fraud})
print("Synthetic transaction data with real-world patterns created.")
print(data[data['TransactionAmount'] > 700].head())  
print(data[data['TransactionAmount'] <= 700].head())  




from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

X = data[['TransactionAmount']]
y = data['Fraud']


model = IsolationForest(contamination=0.05)
model.fit(X)

data['Anomaly'] = model.predict(X)

data['PredictedFraud'] = data['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

accuracy = accuracy_score(y, data['PredictedFraud'])
print(f'IsolationForest model accuracy: {accuracy:.2f}')


import matplotlib.pyplot as plt

plt.scatter(data['TransactionAmount'], data['Fraud'], c=data['Anomaly'], cmap='coolwarm')
plt.xlabel('Transaction Amount')
plt.ylabel('Fraud')
plt.title('Fraud Detection')
plt.show()


def predict_fraud(transaction_amount):
    input_data = pd.DataFrame({'TransactionAmount': [transaction_amount]})
    prediction = model.predict(input_data)
    return prediction[0] == -1

# Example usage
transaction_amount = 500
is_fraud = predict_fraud(transaction_amount)
print(f'The transaction of amount ${transaction_amount} is {"fraudulent" if is_fraud else "not fraudulent"}.')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier()
model.fit(data[['TransactionAmount']], data['Fraud'])

predictions = model.predict(data[['TransactionAmount']])
accuracy = accuracy_score(data['Fraud'], predictions)
print(f'Random Forest model accuracy: {accuracy:.2f}')


import matplotlib.pyplot as plt

# Scatter plot for transaction amounts and fraud labels
plt.figure(figsize=(12, 6))
plt.scatter(data['TransactionAmount'], data['Fraud'], c=predictions, cmap='coolwarm', alpha=0.6)
plt.xlabel('Transaction Amount')
plt.ylabel('Fraud')
plt.title('Fraud Detection Scatter Plot')
plt.show()




plt.figure(figsize=(12, 6))
plt.hist(data[data['Fraud'] == 0]['TransactionAmount'], bins=30, alpha=0.5, label='Not Fraud')
plt.hist(data[data['Fraud'] == 1]['TransactionAmount'], bins=30, alpha=0.5, label='Fraud')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.title('Transaction Amount Distribution')
plt.legend()
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


X = data[['TransactionAmount']]
y = data['Fraud']


model = LogisticRegression()
model.fit(X, y)


predictions = model.predict(X)


accuracy = accuracy_score(y, predictions)
print(f'Logistic Regression model accuracy: {accuracy:.2f}')



from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


X = data[['TransactionAmount']]
y = data['Fraud']


model = SVC()
model.fit(X, y)


predictions = model.predict(X)


accuracy = accuracy_score(y, predictions)
print(f'SVM model accuracy: {accuracy:.2f}')


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


X = data[['TransactionAmount']]
y = data['Fraud']

model = GradientBoostingClassifier()
model.fit(X, y)


predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)
print(f'Gradient Boosting model accuracy: {accuracy:.2f}')

# Isolation Forest: Best for detecting "weird" transactions that don't fit the norm.
# Random Forest: Good all-rounder, works well when you have multiple indicators of fraud.
# Logistic Regression: Simple and fast, good for clear-cut cases.
# SVM: Powerful for complex patterns, but can be slower.
# Gradient Boosting: Often provides the highest accuracy, but can be complex to tune.




