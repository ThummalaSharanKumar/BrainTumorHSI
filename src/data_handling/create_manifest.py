# src/data_handling/create_manifest.py (CORRECTED)
import os
import pandas as pd
import sys

print("--- Starting Phase 1.1: Creating Master Data Manifest ---")

# Define paths relative to the project root
DATA_ROOT = 'data_raw'
OUTPUT_FILE = 'manifest.csv'

if not os.path.isdir(DATA_ROOT):
    print(f"!!! ERROR: Directory '{DATA_ROOT}' not found.")
    print("Please ensure your 'data_raw' folder is in the same directory as this script is run from.")
    sys.exit()

manifest_data = []
capture_folders = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
print(f"Scanning {len(capture_folders)} directories in '{DATA_ROOT}'...")

for folder_name in sorted(capture_folders):
    try:
        patient_id, capture_id = folder_name.split('-')
        nested_path = os.path.join(DATA_ROOT, folder_name, folder_name)
        hsi_path = os.path.join(nested_path, 'raw.hdr')
        gt_path = os.path.join(nested_path, 'gtMap.hdr')

        if os.path.exists(hsi_path) and os.path.exists(gt_path):
            manifest_data.append({
                'patient_id': patient_id, 'capture_id': capture_id,
                'hsi_path': hsi_path, 'gt_path': gt_path
            })
    except ValueError:
        print(f"--- WARNING: Skipping '{folder_name}'. Could not parse.")

if not manifest_data:
    print("!!! ERROR: No valid data entries were found. Check your data_raw folder structure.")
    sys.exit()

manifest_df = pd.DataFrame(manifest_data)
manifest_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n--- Success! Created {OUTPUT_FILE} with {len(manifest_df)} entries. ---")