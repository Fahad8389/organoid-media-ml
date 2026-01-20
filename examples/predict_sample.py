#!/usr/bin/env python3
"""
Example: Make predictions with the trained model.

Usage:
    python examples/predict_sample.py
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from config.paths import DB_PATH, MODEL_ARTIFACTS_DIR, PROCESSED_DATA_DIR
from src.models import MediaRecipeGenerator
from src.preprocessing import MediaRecipePreprocessor
import sqlite3


def main():
    # 1. Load the trained model
    print("Loading model...")
    model_path = MODEL_ARTIFACTS_DIR / "media_recipe_generator.joblib"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run 'python train.py' first to train the model.")
        sys.exit(1)

    model = MediaRecipeGenerator.load(model_path)

    # 2. Load sample data (first 5 rows from training data)
    print("Loading sample data...")
    data_path = PROCESSED_DATA_DIR / "master_dataset_v2.csv"

    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path, nrows=5)

    # 3. Create and fit preprocessor (or load saved one)
    print("Preprocessing...")
    preprocessor = MediaRecipePreprocessor()
    X, y = preprocessor.fit_transform(df)

    # 4. Make predictions
    print("Predicting...")
    predictions = model.predict(X)

    # 5. Format and display results
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    for i, (idx, row) in enumerate(df.iterrows()):
        print(f"\nSample {i+1}: {row.get('model_name', 'Unknown')}")
        print(f"  Primary Site: {row.get('primary_site', 'Unknown')}")
        print(f"  Predicted Media Recipe:")

        for factor, values in predictions.items():
            if i < len(values):
                value = values[i]
                if pd.notna(value):
                    if isinstance(value, float):
                        print(f"    - {factor}: {value:.2f}")
                    else:
                        print(f"    - {factor}: {value}")
                else:
                    print(f"    - {factor}: Not needed")


if __name__ == "__main__":
    main()
