
import pandas as pd
import numpy as np

np.random.seed(42)
size = np.random.randint(500, 4000, size=1000)
location = np.random.choice(['Location_' + str(i) for i in range(1, 11)], size=1000, replace=True)
price = size * np.random.randint(100, 500, size=1000)

data = pd.DataFrame({'Size': size, 'Location': location, 'Price': price})
print("Synthetic real estate data created.")

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.get_dummies(data, columns=['Location'], drop_first=True)
X = data.drop('Price', axis=1)
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Real estate price prediction MSE: {mse:.2f}')


import matplotlib.pyplot as plt

feature_importance = model.coef_
features = X.columns

plt.bar(features, feature_importance)

plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.show()

def predict_price(size, location):
    input_data = pd.DataFrame({'Size': [size], 'Location': [location]})
    input_data = pd.get_dummies(input_data, columns=['Location'], drop_first=True)

    # Add missing columns with value 0
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match the training data
    input_data = input_data[X.columns]

    return model.predict(input_data)


# Example usage
size = 2000
location = 'Location_5'
predicted_price = predict_price(size, location)

print(f'The predicted price for a property of size {size} sqft in {location} is ${predicted_price[0]:.2f}')


import matplotlib.pyplot as plt

def predict_price(sizes, locations):
    if isinstance(sizes, list) and isinstance(locations, list):
        predictions = []
        for size, location in zip(sizes, locations):
            input_data = pd.DataFrame({'Size': [size], 'Location': [location]})
            input_data = pd.get_dummies(input_data, columns=['Location'], drop_first=True)

            # Add missing columns with value 0
            for col in X.columns:
                if col not in input_data.columns:
                    input_data[col] = 0

            # Reorder columns to match the training data
            input_data = input_data[X.columns]

            predictions.append(model.predict(input_data)[0])
        return predictions
    else:
        input_data = pd.DataFrame({'Size': [sizes], 'Location': [locations]})
        input_data = pd.get_dummies(input_data, columns=['Location'], drop_first=True)

        # Add missing columns with value 0
        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder columns to match the training data
        input_data = input_data[X.columns]

        return model.predict(input_data)

# Generate synthetic data for plotting
size = np.linspace(500, 4000, 100).tolist()
location = ['Location_5'] * 100
price = predict_price(size, location)

plt.figure(figsize=(10, 6))
plt.scatter(data['Size'], data['Price'], label='Actual Data', color='blue')

plt.plot(size, price, label='Predicted Price', color='red')
plt.xlabel('Size (sqft)')
plt.ylabel('Price ($)')
plt.title('Real Estate Price Prediction')
plt.legend()

plt.show()

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Real estate price prediction MSE: {mse:.2f}')

import matplotlib.pyplot as plt

feature_importance = model.feature_importances_
features = X.columns

plt.bar(features, feature_importance)

plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.show()


# Example usage
size = 2000
location = 'Location_5'
predicted_price = predict_price(size, location)

print(f'The predicted price for a property of size {size} sqft in {location} is ${predicted_price[0]:.2f}')

size = np.linspace(500, 4000, 100).tolist()
location = ['Location_5'] * 100
price = predict_price(size, location)

plt.figure(figsize=(10, 6))
plt.scatter(data['Size'], data['Price'], label='Actual Data', color='blue')

plt.plot(size, price, label='Predicted Price', color='red')
plt.xlabel('Size (sqft)')
plt.ylabel('Price ($)')
plt.title('Real Estate Price Prediction')
plt.legend()

plt.show()

