# README.md

# Two-Layer Fruit Ripeness Detection System

A complete machine learning system for detecting fruit type and ripeness level using RGB color values. Designed for deployment on Raspberry Pi integration for real-time LED indication.

## System Overview

This project implements a **two-layer detection pipeline**:

1. **Layer 1: Fruit Type Classification** - Identifies which fruit (Apple, Banana, Mango, Orange) is being detected
2. **Layer 2: Ripeness Classification** - Determines ripeness stage (Early Ripe, Partially Ripe, Ripe, Decay) specific to the detected fruit

## Architecture

```
RGB Color Input (R, G, B)
         ↓
    ┌────────────────────────────┐
    │  LAYER 1: Fruit Detector   │
    │  (fruit_classifier.py)     │
    │  Neural Network (3-32-16-5)│
    └────────────────────────────┘
         ↓ (Fruit Type)
         ↓ (e.g., Banana)
    ┌────────────────────────────┐
    │ LAYER 2: Ripeness Detector │
    │(train_ripeness_per_fruit.py)│
    │ 5 Fruit-Specific Models    │
    │ Each: (3-16-8-4 neurons)   │
    └────────────────────────────┘
         ↓ (Ripeness Code)
         ↓ (0-3)
    ┌────────────────────────────┐
    │ OUTPUT: Ripeness Level     │
    │ 0 = Early Ripe             │
    │ 1 = Partially Ripe         │
    │ 2 = Ripe                   │
    │ 3 = Decay                  │
    └────────────────────────────┘
         ↓
    ┌────────────────────────────┐
    │   LED Control      │
    │   Pin 2: Green  (0)        │
    │   Pin 3: Yellow (1)        │
    │   Pin 4: White  (2)        │
    │   Pin 5: Red    (3)        │
    └────────────────────────────┘
```

## Project Files

### Core Training Scripts

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `fruit_classifier.py` | Train fruit type detector | CSV (R,G,B,fruit_type) | `fruit_classifier_model.joblib` |
| `train_ripeness_per_fruit.py` | Train ripeness models per fruit | CSV (R,G,B,fruit_type,ripeness) | `ripeness_models/` (5 models) |
| `main_ripeness_detector.py` | End-to-end inference | RGB values | Fruit + Ripeness prediction |

### Documentation

| File | Content |
|------|---------|
| `SETUP_AND_TRAINING_GUIDE.md` | Complete installation, training, and deployment guide |
| `QUICK_START.txt` | Command reference and quick setup |
| `README.md` | This file - project overview and architecture |

### Data Files

| File | Description |
|------|-------------|
| `fruit_ripeness_dataset.csv` | Synthetic dataset (1000 samples, 4 fruits, 4 ripeness stages) |
| `demo_rgb_data.csv` | Small demo dataset for testing |

## Getting Started

### 1. Installation

```bash
# Create virtual environment
python -m venv proj_sklearn-env

# Activate environment
proj_sklearn-env\Scripts\activate.bat  # Windows
# or
source proj_sklearn-env/bin/activate  # Linux/Mac

# Install dependencies
pip install numpy pandas scikit-learn joblib
```

### 2. Training

```bash
# Train Layer 1: Fruit Classifier
python fruit_classifier.py --csv fruit_ripeness_dataset.csv --epochs 200

# Train Layer 2: Ripeness Models
python train_ripeness_per_fruit.py --csv fruit_ripeness_dataset.csv --epochs 200
```

### 3. Testing

```bash
# Single prediction
python main_ripeness_detector.py --rgb-input 150 120 85

# Batch testing
python main_ripeness_detector.py --csv-input fruit_ripeness_dataset.csv

# Interactive mode
python main_ripeness_detector.py
```

## Dataset Format

### Training CSV Structure

```csv
R,G,B,fruit_type,ripeness_label
120,80,60,0,0
200,140,90,2,2
150,120,85,1,2
...
```

**Columns:**
- `R, G, B`: RGB color values (0-255)
- `fruit_type`: 0=Apple, 1=Banana, 2=Mango, 3=Orange
- `ripeness_label`: 0=Early Ripe, 1=Partially Ripe, 2=Ripe, 3=Decay

## Label Mappings

### Fruit Types
```
0 = Apple
1 = Banana
2 = Mango
3 = Orange
```

### Ripeness Stages (Arduino LED Codes)
```
0 = Early Ripe      → Green LED
1 = Partially Ripe  → Yellow LED
2 = Ripe           → White LED
3 = Decay          → Red LED
```

## Model Architecture

### Layer 1: Fruit Classifier
- **Input**: 3 neurons (R, G, B)
- **Hidden Layer 1**: 32 neurons (ReLU activation)
- **Hidden Layer 2**: 16 neurons (ReLU activation)
- **Output**: 4 neurons (softmax - one per fruit type)
- **Activation**: ReLU for hidden layers
- **Optimizer**: Adam
- **Loss**: Categorical Cross-entropy

### Layer 2: Ripeness Models (×5, one per fruit)
- **Input**: 3 neurons (R, G, B)
- **Hidden Layer 1**: 16 neurons (ReLU activation)
- **Hidden Layer 2**: 8 neurons (ReLU activation)
- **Output**: 4 neurons (softmax - one per ripeness stage)
- **Activation**: ReLU for hidden layers
- **Optimizer**: Adam
- **Loss**: Categorical Cross-entropy

## Expected Performance

On Synthetic Dataset:
- **Fruit Type Detection**: 95-99% accuracy
- **Ripeness Prediction**: 90-95% accuracy

On Real Sensor Data:
- Depends on sensor calibration, lighting conditions, and training data quality
- Recommended: Collect 50+ samples per fruit per ripeness stage

## Raspberry Pi Deployment

### Requirements
- Raspberry Pi 3 or higher
- 4 LED connections
- RGB color sensor (e.g., TCS34725)
- Serial communication cable

### Installation on Raspberry Pi

```bash
pip install numpy pandas scikit-learn joblib pyserial

# Copy model files to Raspberry Pi
scp fruit_classifier_model.joblib pi@raspberrypi:~/fruit_detector/
scp -r ripeness_models pi@raspberrypi:~/fruit_detector/
scp main_ripeness_detector.py pi@raspberrypi:~/fruit_detector/
```

### Arduino Integration


## Key Features

✅ **Two-Layer Architecture**: Fruit detection followed by ripeness classification
✅ **Multi-Fruit Support**: Handles 4 different fruit types with specific color models
✅ **Lightweight**: Suitable for Raspberry Pi deployment
✅ **Integration**: Direct LED output control
✅ **Flexible Inference**: Single input, batch processing, or interactive modes
✅ **Comprehensive Documentation**: Setup guides and quick references
✅ **Reproducible**: Fixed random seeds for consistent results
✅ **Validated**: Per-class precision/recall/F1 metrics and confusion matrices


## Customization

### Adding New Fruits

1. Create training data with new fruit type ID (4+)
2. Update `FRUIT_MAP` in all three scripts
3. Retrain both layers with new data

### Adjusting Model Complexity

Edit hidden layer sizes in `build_model()` functions:
- Smaller networks: Faster inference, less accurate
- Larger networks: Slower inference, more accurate

### Changing Ripeness Stages

Update `RIPENESS_MAP` and redefine ripeness labels (currently 0-3, can be expanded to 0-5)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "ModuleNotFoundError" | Run `pip install numpy pandas scikit-learn joblib` |
| "FileNotFoundError: CSV" | Ensure CSV file is in same directory as scripts |
| "Model not found" | Run training scripts first |
| "Poor accuracy" | Collect more real-world training data |
| Serial connection fails | Check Arduino port and baud rate settings |

## Performance Optimization for Raspberry Pi

1. **Model Quantization**: Convert to ONNX or TensorFlow Lite
2. **Batch Prediction**: Process multiple samples simultaneously
3. **Caching**: Store frequently used predictions
4. **Threading**: Process sensor input in background thread

## References

- Original Project: [Ripeness-Detector-for-Vegetables-and-Fruits](https://github.com/jayant1211/Ripeness-Detector-for-Vegetables-and-Fruits)
- IEEE Publication: Ripeness Detection by Spectral Analysis (CONIT 2022)
- Machine Learning Framework: scikit-learn
- Data Processing: Pandas, NumPy


## Next Steps

1. ✅ Install and run on your system
2. ✅ Train models with provided synthetic dataset
3. ✅ Collect real sensor data
4. ✅ Retrain with real data for production accuracy
5. ✅ Deploy on Raspberry Pi
6. ✅ Monitor and log predictions for continuous improvement


---

**Version**: 2.0 (Two-layer detection system)
