import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('house_prices.txt')

print(df.head())

df.dropna(inplace=True)

features = df[['sqft', 'bedrooms', 'bathrooms']]
target = df['price']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse_rounded = round(mse, 2)
r2_rounded = round(r2, 2)

print(f"Mean Squared Error: {mse_rounded}")
print(f"R^2 Score: {r2_rounded}")

# Example prediction
# Predict the price of a house with 2000 sqft, 3 bedrooms, and 2 bathrooms
example_house = np.array([[2000, 3, 2]])
predicted_price = regressor.predict(example_house)

predicted_price_rounded = round(predicted_price[0], 2)

print(f"Predicted Price for a house with 2000 sqft, 3 bedrooms, and 2 bathrooms: {predicted_price_rounded}")