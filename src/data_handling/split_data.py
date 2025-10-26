# src/data_handling/split_data.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

print("--- Starting Phase 1.2: Patient-Aware Data Splitting ---")

# --- 1. DEFINE PATHS AND SPLIT RATIOS ---
MANIFEST_FILE = os.path.join('..', 'manifest.csv')
OUTPUT_DIR = os.path.join('..') # Save to the root project folder

# We will use an 70% / 15% / 15% split for Train / Validation / Test
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# --- 2. LOAD THE MASTER MANIFEST ---
if not os.path.exists(MANIFEST_FILE):
    print(f"!!! ERROR: '{MANIFEST_FILE}' not found. Please run 'create_manifest.py' first.")
    exit()

manifest_df = pd.read_csv(MANIFEST_FILE)
print(f"Loaded master manifest with {len(manifest_df)} total captures.")

# --- 3. PERFORM THE PATIENT-WISE SPLIT ---
# Get a list of all unique patient IDs
patient_ids = manifest_df['patient_id'].unique()
print(f"Found {len(patient_ids)} unique patients.")

# Split the unique patient IDs into training and a temporary set (validation + test)
train_patients, temp_patients = train_test_split(
    patient_ids,
    test_size=(VALIDATION_RATIO + TEST_RATIO),
    random_state=42  # Using a fixed random_state ensures the split is the same every time
)

# Split the temporary set into validation and test sets
# We need to adjust the test_size ratio for the second split
val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=(TEST_RATIO / (VALIDATION_RATIO + TEST_RATIO)),
    random_state=42
)

print("\nSplitting patients into sets:")
print(f"- Training Set: {len(train_patients)} patients")
print(f"- Validation Set: {len(val_patients)} patients")
print(f"- Test Set: {len(test_patients)} patients")

# --- 4. CREATE THE FINAL DATAFRAMES ---
# Filter the original manifest to create a new DataFrame for each set
train_df = manifest_df[manifest_df['patient_id'].isin(train_patients)]
val_df = manifest_df[manifest_df['patient_id'].isin(val_patients)]
test_df = manifest_df[manifest_df['patient_id'].isin(test_patients)]

print("\nSplitting captures into sets:")
print(f"- Training Set: {len(train_df)} captures")
print(f"- Validation Set: {len(val_df)} captures")
print(f"- Test Set: {len(test_df)} captures")

# --- 5. SAVE THE SPLIT FILES ---
train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)

print("\n--- Success! ---")
print("Created 'train.csv', 'val.csv', and 'test.csv' in the project root directory.")
print("\nSample of train.csv:")
print(train_df.head())