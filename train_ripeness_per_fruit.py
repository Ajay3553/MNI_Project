"""
Purpose:
  - Implements Step C: Train ripeness detection models for each fruit type separately.
  - Uses separate MLP models for ripeness classification per fruit.
  - Ensures stable training and prevents biased "always Decay" output.

Expected CSV schema:
  Columns: R, G, B, fruit_type, ripeness_label
  R,G,B ∈ [0,255]; fruit_type ∈ {0..4}; ripeness_label ∈ {0..3}

Fruit type mapping:
  0 = Apple
  1 = Banana
  2 = Mango
  3 = Orange

Ripeness label mapping:
  0 = Early Ripe
  1 = Partially Ripe
  2 = Ripe
  3 = Decay
"""

import argparse
import os
import random
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Fixed mappings
FRUIT_MAP: Dict[int, str] = {
    0: "Apple",
    1: "Banana",
    2: "Mango",
    3: "Orange",
}

RIPENESS_MAP: Dict[int, str] = {
    0: "Early Ripe",
    1: "Partially Ripe",
    2: "Ripe",
    3: "Decay"
}


# ---------------- Utility Functions ---------------- #

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = ["R", "G", "B", "fruit_type", "ripeness_label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()
    for ch in ["R", "G", "B"]:
        df[ch] = pd.to_numeric(df[ch], errors="coerce").clip(0, 255)

    df["fruit_type"] = pd.to_numeric(df["fruit_type"], errors="coerce")
    df["ripeness_label"] = pd.to_numeric(df["ripeness_label"], errors="coerce")

    mask = (
        df["R"].between(0, 255)
        & df["G"].between(0, 255)
        & df["B"].between(0, 255)
        & df["fruit_type"].isin(FRUIT_MAP.keys())
        & df["ripeness_label"].isin(RIPENESS_MAP.keys())
    )
    dropped_df = df.loc[~mask]
    clean_df = df.loc[mask].copy()

    clean_df[["R", "G", "B", "fruit_type", "ripeness_label"]] = clean_df[
        ["R", "G", "B", "fruit_type", "ripeness_label"]
    ].astype(int)

    return clean_df, dropped_df


def print_class_counts(df: pd.DataFrame, header: str, fruit_id: Optional[int] = None) -> None:
    print(f"\n{header}")
    if fruit_id is not None:
        for ripeness_id in sorted(RIPENESS_MAP.keys()):
            count = int((df["ripeness_label"] == ripeness_id).sum())
            print(f"  Ripeness {ripeness_id} ({RIPENESS_MAP[ripeness_id]}): {count}")
    else:
        for fruit_id in sorted(FRUIT_MAP.keys()):
            count = int((df["fruit_type"] == fruit_id).sum())
            print(f"  Fruit {fruit_id} ({FRUIT_MAP[fruit_id]}): {count}")


# ---------------- Core Training Functions ---------------- #

def split_and_scale(df: pd.DataFrame, val_split: float, seed: int):
    X = df[["R", "G", "B"]].values
    y = df["ripeness_label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, stratify=y, random_state=seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print(f"  Split sizes → Train: {len(X_train)}, Val: {len(X_val)}")
    return X_train_scaled, X_val_scaled, y_train, y_val, scaler


def build_model(seed: int, hidden=(32, 16), epochs=300) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate="adaptive",
        max_iter=epochs,
        random_state=seed,
        verbose=False
    )


def train_and_evaluate(model, X_train, y_train, X_val, y_val, fruit_name):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    print(f"\n  ===== {fruit_name} Validation =====")
    print(f"  Accuracy: {acc:.4f}")
    print(
        classification_report(
            y_val,
            preds,
            labels=sorted(RIPENESS_MAP.keys()),
            target_names=[RIPENESS_MAP[i] for i in sorted(RIPENESS_MAP.keys())],
            digits=3,
            zero_division=0,
        )
    )
    return acc


# ---------------- Save Artifacts ---------------- #

def save_artifacts(models_dict: Dict[int, Dict[str, object]], model_dir: str, metadata: Dict[str, object]) -> None:
    os.makedirs(model_dir, exist_ok=True)
    for fruit_id, data in models_dict.items():
        fruit_name = FRUIT_MAP[fruit_id]
        out_path = os.path.join(model_dir, f"ripeness_model_{fruit_name.lower()}.joblib")
        payload = {
            "model": data["model"],
            "scaler": data["scaler"],
            "fruit_id": fruit_id,
            "fruit_name": fruit_name,
            "ripeness_map": RIPENESS_MAP,
            "metadata": metadata
        }
        joblib.dump(payload, out_path)
        print(f"✅ Saved model for {fruit_name} → {out_path}")


# ---------------- Main ---------------- #

def main():
    parser = argparse.ArgumentParser(description="Train per-fruit ripeness models (Step C)")
    parser.add_argument("--csv", default="data/fruit_ripeness_dataset.csv", help="Input CSV path")
    parser.add_argument("--model-dir", default="ripeness_models", help="Output model directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    set_global_seed(args.seed)

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df_raw = pd.read_csv(args.csv)
    df, dropped = validate_and_clean(df_raw)
    print(f"\nLoaded {len(df_raw)} rows → Valid: {len(df)}, Dropped: {len(dropped)}")
    print_class_counts(df, "Overall fruit type distribution:")

    models_dict = {}
    for fruit_id in sorted(FRUIT_MAP.keys()):
        fruit_name = FRUIT_MAP[fruit_id]
        print(f"\n{'='*60}")
        print(f"Training model for {fruit_name} (ID: {fruit_id})")
        print(f"{'='*60}")

        df_fruit = df[df["fruit_type"] == fruit_id]
        if df_fruit.empty:
            print(f"⚠️ No data for {fruit_name}, skipping...")
            continue

        print_class_counts(df_fruit, f"  Ripeness distribution for {fruit_name}:", fruit_id=fruit_id)
        X_train, X_val, y_train, y_val, scaler = split_and_scale(df_fruit, args.val_split, args.seed)

        model = build_model(args.seed, hidden=(32, 16), epochs=args.epochs)
        acc = train_and_evaluate(model, X_train, y_train, X_val, y_val, fruit_name)

        models_dict[fruit_id] = {"model": model, "scaler": scaler}

    metadata = {
        "seed": args.seed,
        "val_split": args.val_split,
        "epochs": args.epochs,
        "columns": ["R", "G", "B"],
        "range": [0, 255],
        "ripeness_map": RIPENESS_MAP,
    }

    save_artifacts(models_dict, args.model_dir, metadata)
    print("\n✅ All ripeness models trained and saved successfully!\n")


if __name__ == "__main__":
    main()
