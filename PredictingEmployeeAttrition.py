# Day 6 - Predicting Employee Attrition

# pip install pandas scikit-learn matplotlib wordcloud plotly seaborn statsmodels numpy


# ------------------ Synthetic Data ------------------
import pandas as pd
import numpy as np

np.random.seed(42)
age = np.random.randint(20, 60, size=1000)
salary = np.random.randint(30000, 120000, size=1000)
years_at_company = np.random.randint(1, 20, size=1000)
attrition = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])

data = pd.DataFrame({'Age': age, 'Salary': salary, 'YearsAtCompany': years_at_company, 'Attrition': attrition})
print("Synthetic employee attrition data created.")


# ------------------ Modeling ------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = data[['Age', 'Salary', 'YearsAtCompany']]
y = data['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Employee attrition prediction accuracy: {accuracy:.2f}')

# ------------------ Visualize the Decision Tree ------------------
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=['No Attrition', 'Attrition'], filled=True)
plt.show()


# ------------------ Feature Importance ------------------
import matplotlib.pyplot as plt

feature_importance = model.feature_importances_
features = X.columns

plt.bar(features, feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')

plt.show()

# ------------------ Predictions ------------------
def predict_attrition(age, salary, years_at_company, model):
    input_data = pd.DataFrame({'Age': [age], 'Salary': [salary], 'YearsAtCompany': [years_at_company]})
    prediction = model.predict(input_data)[0]
    return prediction

# Example usage
age = 35
salary = 20000
years_at_company = 10
prediction = predict_attrition(age, salary, years_at_company, model)
print(f'Predicted attrition for an employee with age {age}, salary {salary}, and {years_at_company} years at the company: {prediction}')

# ------------------ Recommendations ------------------
def recommend_training(age, salary, years_at_company, model):
    input_data = pd.DataFrame({'Age': [age], 'Salary': [salary], 'YearsAtCompany': [years_at_company]})
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return 'Employee is at risk of attrition. Recommend training programs to improve employee satisfaction.'
    else:
        return 'Employee is not at risk of attrition. No training programs recommended.'

# Example usage
age = 25
salary = 20000
years_at_company = 5
recommendation = recommend_training(age, salary, years_at_company, model)
print(recommendation)




