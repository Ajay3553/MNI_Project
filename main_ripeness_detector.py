#!/usr/bin/env python3
"""
Two-Layer Fruit Ripeness Detection System
==========================================
Layer 1: Detects fruit type using TensorFlow CNN (fruit_classifier_model.h5)
Layer 2: Predicts ripeness based on RGB values using fruit-specific MLP models

Usage:
  python main_ripeness_detector.py --image test.jpg
  python main_ripeness_detector.py --rgb-input 150 120 85 --fruit-id 0
  python main_ripeness_detector.py  (uses camera)
"""

import argparse
import os
import sys
from typing import Tuple, Dict
import joblib
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

# Fixed label mappings (4 fruits now, matching your trained model)
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

class FruitRipenessDetector:
    """Two-layer detection: CNN for fruit type + MLP for ripeness"""
    def __init__(self, fruit_model_path: str, ripeness_models_dir: str):
        self.fruit_model_path = fruit_model_path
        self.ripeness_models_dir = ripeness_models_dir

        # Load Layer 1: TensorFlow fruit classifier
        if not os.path.exists(fruit_model_path):
            raise FileNotFoundError(f"Fruit classifier model not found: {fruit_model_path}")
        self.fruit_model = keras.models.load_model(fruit_model_path)
        
        # Detect input size from model
        self.img_size = self.fruit_model.input_shape[1]  # Auto-detect (224 or 128)
        print(f"✓ Loaded fruit classifier from: {fruit_model_path} (Input size: {self.img_size}x{self.img_size})")

        # Load Layer 2: Ripeness models per fruit
        self.ripeness_models = {}
        self.ripeness_scalers = {}
        if os.path.exists(ripeness_models_dir):
            for fruit_id, fruit_name in FRUIT_MAP.items():
                model_path = os.path.join(ripeness_models_dir, f"ripeness_model_{fruit_name.lower()}.joblib")
                if os.path.exists(model_path):
                    artifacts = joblib.load(model_path)
                    self.ripeness_models[fruit_id] = artifacts["model"]
                    self.ripeness_scalers[fruit_id] = artifacts["scaler"]
                    print(f"✓ Loaded ripeness model for {fruit_name}")
                else:
                    print(f"⚠ Ripeness model not found for {fruit_name}: {model_path}")
        else:
            print(f"⚠ Ripeness models directory not found: {ripeness_models_dir}")

    def predict_fruit_type_from_image(self, image: np.ndarray) -> Tuple[int, str, float]:
        """Layer 1: Predict fruit type from image using CNN"""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))  # Auto-resize based on model
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        pred = self.fruit_model.predict(img, verbose=0)
        fruit_id = int(np.argmax(pred[0]))
        confidence = float(pred[0][fruit_id])
        return fruit_id, FRUIT_MAP[fruit_id], confidence

    def predict_ripeness_from_rgb(self, rgb: np.ndarray, fruit_id: int) -> Tuple[int, str]:
        """Layer 2: Predict ripeness from RGB using fruit-specific MLP"""
        if fruit_id not in self.ripeness_models:
            raise ValueError(f"No ripeness model for fruit ID {fruit_id}")
        
        x = rgb.reshape(1, -1).astype(np.float32)
        x_scaled = self.ripeness_scalers[fruit_id].transform(x)
        ripeness_id = int(self.ripeness_models[fruit_id].predict(x_scaled)[0])
        return ripeness_id, RIPENESS_MAP[ripeness_id]

    def detect_from_image(self, image: np.ndarray) -> Dict[str, any]:
        """Full pipeline: image → fruit type → RGB → ripeness"""
        # Layer 1: Detect fruit type
        fruit_id, fruit_name, confidence = self.predict_fruit_type_from_image(image)
        
        # Extract average RGB
        avg_color = cv2.mean(image)[:3]  # (B, G, R)
        b, g, r = map(int, avg_color)
        rgb = np.array([r, g, b], dtype=np.float32)
        
        # Layer 2: Detect ripeness
        ripeness_id, ripeness_name = self.predict_ripeness_from_rgb(rgb, fruit_id)
        
        return {
            "fruit_id": fruit_id,
            "fruit_name": fruit_name,
            "fruit_confidence": confidence,
            "ripeness_id": ripeness_id,
            "ripeness_name": ripeness_name,
            "arduino_code": ripeness_id,
            "rgb": (r, g, b)
        }

    def detect_from_rgb(self, r: int, g: int, b: int, fruit_id: int) -> Dict[str, any]:
        """Manual mode: RGB + fruit_id → ripeness"""
        rgb = np.array([r, g, b], dtype=np.float32)
        ripeness_id, ripeness_name = self.predict_ripeness_from_rgb(rgb, fruit_id)
        return {
            "fruit_id": fruit_id,
            "fruit_name": FRUIT_MAP[fruit_id],
            "fruit_confidence": 1.0,
            "ripeness_id": ripeness_id,
            "ripeness_name": ripeness_name,
            "code": ripeness_id,
            "rgb": (r, g, b)
        }

def print_result(result: Dict[str, any]):
    print("\n" + "="*60)
    print("RIPENESS DETECTION RESULT")
    print("="*60)
    print(f"Fruit Type:        {result['fruit_name']} (ID: {result['fruit_id']})")
    print(f"Confidence:        {result['fruit_confidence']:.4f} ({result['fruit_confidence']*100:.2f}%)")
    print(f"RGB Values:        R={result['rgb'][0]}, G={result['rgb'][1]}, B={result['rgb'][2]}")
    print(f"Ripeness Stage:    {result['ripeness_name']} (Code: {result['ripeness_id']})")
    print(f"Code:              {result['arduino_code']}")
    print("="*60)
    print("\nMapping:")
    print("0 = Green (Early Ripe)")
    print("1 = Yellow (Partially Ripe)")
    print("2 = White (Ripe)")
    print("3 = Red (Decay)")
    print("="*60 + "\n")

def capture_image_from_camera() -> np.ndarray:
    print("📸 Opening camera... Press 'Space' to capture, 'Esc' to exit.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check your connection.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error. Try again.")
            continue
        cv2.imshow("Camera - Press Space to Capture", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            print("❌ Exiting camera.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == 32:  # Space
            print("✅ Captured image.")
            cap.release()
            cv2.destroyAllWindows()
            return frame

def main():
    parser = argparse.ArgumentParser(description="Two-Layer Fruit Ripeness Detector")
    parser.add_argument("--fruit-model", default="fruit_classifier_model.h5")
    parser.add_argument("--ripeness-dir", default="ripeness_models")
    parser.add_argument("--rgb-input", nargs=3, type=int, default=None, help="Manual RGB input (R G B)")
    parser.add_argument("--fruit-id", type=int, default=None, help="Fruit ID (0=Apple, 1=Banana, 2=Mango, 3=Orange)")
    parser.add_argument("--image", type=str, default=None, help="Path to image file for testing")
    args = parser.parse_args()

    detector = FruitRipenessDetector(args.fruit_model, args.ripeness_dir)

    if args.rgb_input:
        if args.fruit_id is None:
            print("ERROR: --fruit-id required when using --rgb-input")
            print("  0 = Apple, 1 = Banana, 2 = Mango, 3 = Orange")
            sys.exit(1)
        r, g, b = args.rgb_input
        print(f"\nTesting with RGB input: R={r}, G={g}, B={b}, Fruit ID={args.fruit_id}")
        result = detector.detect_from_rgb(r, g, b, args.fruit_id)
        print_result(result)

    elif args.image:
        if not os.path.exists(args.image):
            print(f"ERROR: Image not found: {args.image}")
            sys.exit(1)
        image = cv2.imread(args.image)
        result = detector.detect_from_image(image)
        print_result(result)

    else:
        # Live camera mode
        image = capture_image_from_camera()
        result = detector.detect_from_image(image)
        print_result(result)

if __name__ == "__main__":
    main()
