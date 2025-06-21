import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Load test data
X_test = np.load("data/processed/X_processed.npy")
y_test = np.load("data/processed/y_processed.npy")

# Load model
model = tf.keras.models.load_model("models/yield_prediction/saved_model/")

# Predict
y_pred = model.predict(X_test)

# Plot residuals
plt.figure(figsize=(10, 6))
sns.histplot(y_test - y_pred.flatten(), bins=30, kde=True)
plt.title("Prediction Residuals")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Count")
plt.savefig("models/yield_prediction/residuals.png")

# Print metrics
loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MAE: {mae:.2f} tons")
print(f"Test MSE: {mse:.2f}")
