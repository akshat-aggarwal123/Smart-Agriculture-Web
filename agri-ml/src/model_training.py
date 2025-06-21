import os
import numpy as np
from yield_model import build_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tqdm.keras import TqdmCallback
import tensorflow as tf

# ========== Step 1: Create Directories ==========
os.makedirs("models/yield_prediction/saved_model", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ========== Step 2: Load Processed Data ==========
print("📥 Loading processed data...")
X = np.load("data/processed/X_processed.npy")
y = np.load("data/processed/y_processed.npy")

# ========== Step 3: Build Model ==========
print("🧠 Building model...")
model = build_model(X.shape[1])

# ========== Step 4: Train Model ==========
print("🚀 Starting training...")
history = model.fit(
    X, y,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    callbacks=[
        EarlyStopping(patience=15, restore_best_weights=True),
        TensorBoard(log_dir="logs"),
        TqdmCallback(verbose=1)
    ],
    verbose=0  # suppress default TF logging to use tqdm cleanly
)

# ========== Step 5: Save Model ==========
print("💾 Saving model...")
model.save("models/yield_prediction/saved_model/")

# ========== Step 6: Save Metrics ==========
final_mae = history.history['val_mae'][-1]
final_mse = history.history['val_mse'][-1]

with open("models/yield_prediction/metrics.txt", "w") as f:
    f.write(f"Final MAE: {final_mae:.2f} tons\n")
    f.write(f"Final MSE: {final_mse:.2f}\n")

print("✅ Training complete. Model and metrics saved.")
