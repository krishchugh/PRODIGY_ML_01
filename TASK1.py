import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Reading the dataset from the text file
df = pd.read_csv('house_prices.txt')

# Display the first few rows of the dataframe
print(df.head())

# Data preprocessing
# Drop rows with any missing values
df.dropna(inplace=True)

# Define the features (independent variables) and the target (dependent variable)
features = df[['sqft', 'bedrooms', 'bathrooms']]
target = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
regressor = LinearRegression()

# Train the model on the training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Calculate the mean squared error and R^2 score for model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Round off the results
mse_rounded = round(mse, 2)
r2_rounded = round(r2, 2)

print(f"Mean Squared Error: {mse_rounded}")
print(f"R^2 Score: {r2_rounded}")

# Example prediction
# Predict the price of a house with 2000 sqft, 3 bedrooms, and 2 bathrooms
example_house = np.array([[2000, 3, 2]])
predicted_price = regressor.predict(example_house)

# Round off the predicted price
predicted_price_rounded = round(predicted_price[0], 2)

print(f"Predicted Price for a house with 2000 sqft, 3 bedrooms, and 2 bathrooms: {predicted_price_rounded}")
