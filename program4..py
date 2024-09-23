import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Generate synthetic data
np.random.seed(0)
x = np.linspace(-1, 1, 100)  # Input features
y = 3 * x + np.random.normal(0, 0.1, x.shape)  # Linear relationship with some noise

# Split the data into training and testing sets
x_train, x_test = x[:80], x[80:]
y_train, y_test = y[:80], y[80:]

# Build the regression model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(1,)),  # Input layer
    layers.Dense(64, activation='relu'),                    # Hidden layer
    layers.Dense(1)                                         # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(x_train, y_train, epochs=100, verbose=0)

# Evaluate the model
test_loss = model.evaluate(x_test, y_test, verbose=0)
print(f'Test Loss (MSE): {test_loss:.4f}')

# Make predictions
y_pred = model.predict(x_test)

# Plot the training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot predictions vs actual values
plt.subplot(1, 2, 2)
plt.scatter(x, y, label='Data', color='blue')
plt.scatter(x_test, y_pred, label='Predictions', color='red')
plt.title('Regression Model Predictions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

plt.tight_layout()
plt.show()
