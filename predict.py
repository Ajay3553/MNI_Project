import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import argparse

MODEL_PATH = 'fruit_classifier_model.h5'
IMG_SIZE = 128

def predict_image(image_path):
    model = keras.models.load_model(MODEL_PATH)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    pred = model.predict(img, verbose=0)
    top2_idx = np.argsort(pred[0])[-2:][::-1]
    
    classes = ['Apple', 'Banana', 'Mango', 'Orange']
    print(f"Top 1: {classes[top2_idx[0]]} ({pred[0][top2_idx[0]]:.4f})")
    print(f"Top 2: {classes[top2_idx[1]]} ({pred[0][top2_idx[1]]:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    predict_image(args.image)
