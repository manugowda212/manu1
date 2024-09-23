# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing dataset
boston = fetch_openml(name='boston', version=1, as_frame=True)
X = boston.data[['RM']]  # Using only the 'RM' feature (average number of rooms) for easier plotting
y = boston.target  # Target (House prices)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Linear Regression model
reg = LinearRegression()

# Train the model
reg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = reg.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot the regression line and data points
plt.figure(figsize=(10, 6))

# Plot training data
plt.scatter(X_train, y_train, color='blue', label='Training data')
# Plot testing data
plt.scatter(X_test, y_test, color='green', label='Testing data')

# Plot the regression line (on training data)
plt.plot(X_test, y_pred, color='red', label='Regression line')

# Add labels and title
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('House Price')
plt.title('Linear Regression: Number of Rooms vs House Price')
plt.legend()

# Show the plot
plt.show()

# Residuals Plot (for error analysis)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred - y_test, color='purple')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Residuals (Predictions - True Values)')
plt.title('Residuals Plot')
plt.show()


