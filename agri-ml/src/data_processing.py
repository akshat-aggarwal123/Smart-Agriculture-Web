import pandas as pd
import numpy as np
import joblib
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from scipy import sparse  # for saving sparse matrix

# Wrap tqdm around pandas read_csv
print("Loading datasets...")
farmer_df = pd.read_csv("data/raw/farmer_advisor_dataset.csv")
market_df = pd.read_csv("data/raw/market_researcher_dataset.csv")

# Create directories
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models/yield_prediction", exist_ok=True)

# Merge datasets on common key (adjust column names as needed)
print("Merging datasets...")
merged_df = pd.merge(
    farmer_df,
    market_df,
    left_on="Crop_Type",
    right_on="Product",
    how="inner"
)

# Drop irrelevant columns
print("Dropping irrelevant columns...")
merged_df = merged_df.drop(["Farm_ID", "Market_ID", "Product"], axis=1)

# Handle missing values
print("Handling missing values...")
merged_df = merged_df.fillna(merged_df.median(numeric_only=True))

# Define features and target
X = merged_df.drop("Crop_Yield_ton", axis=1)
y = merged_df["Crop_Yield_ton"]

# Identify feature types
numeric_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(exclude=np.number).columns

# Create preprocessing pipeline
print("Preprocessing features...")
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Apply preprocessing
print("Applying transformation...")
X_processed = preprocessor.fit_transform(X)  # This is a sparse matrix

# Save preprocessor
print("Saving preprocessor...")
joblib.dump(preprocessor, "models/yield_prediction/preprocessor.joblib")

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Save processed data
print("Saving processed data...")
# Save X_processed as sparse
if sparse.issparse(X_processed):
    sparse.save_npz("data/processed/X_processed.npz", X_processed)
    print("✅ Saved sparse matrix using `save_npz`.")
else:
    X_processed = np.asarray(X_processed)  # ensures type safety
    np.save("data/processed/X_processed.npy", X_processed)
    print("✅ Saved dense matrix using `np.save`.")

# Save y (target) as dense
np.save("data/processed/y_processed.npy", y.to_numpy())

print("Preprocessing complete ✅")
