import pandas as pd
import numpy as np

np.random.seed(42)

def generate_samples(base_r, base_g, base_b, var_r, var_g, var_b, count, fruit_type, ripeness):
    """Generate realistic RGB samples with controlled variation"""
    samples = []
    for i in range(count):
        r = int(np.clip(np.random.normal(base_r, var_r), 0, 255))
        g = int(np.clip(np.random.normal(base_g, var_g), 0, 255))
        b = int(np.clip(np.random.normal(base_b, var_b), 0, 255))
        samples.append([r, g, b, fruit_type, ripeness])
    return samples

# Initialize dataset
data = []

# ========== APPLE (fruit_type=0) ==========
# Early Ripe (0): Green-Yellow
data.extend(generate_samples(130, 150, 75, 15, 20, 12, 100, 0, 0))
# Partially Ripe (1): Yellow-Red transition
data.extend(generate_samples(170, 100, 65, 10, 12, 10, 100, 0, 1))
# Ripe (2): Bright Red
data.extend(generate_samples(205, 60, 45, 12, 12, 10, 100, 0, 2))
# Decay (3): Brown-Dark
data.extend(generate_samples(135, 75, 55, 10, 10, 10, 100, 0, 3))

# ========== BANANA (fruit_type=1) ==========
# Early Ripe (0): Green
data.extend(generate_samples(120, 140, 70, 12, 15, 12, 100, 1, 0))
# Partially Ripe (1): Yellow-Green
data.extend(generate_samples(185, 155, 50, 10, 10, 8, 100, 1, 1))
# Ripe (2): Bright Yellow
data.extend(generate_samples(235, 200, 40, 10, 10, 6, 100, 1, 2))
# Decay (3): Brown with spots
data.extend(generate_samples(145, 120, 48, 10, 10, 6, 100, 1, 3))

# ========== MANGO (fruit_type=2) ==========
# Early Ripe (0): Green
data.extend(generate_samples(125, 150, 75, 15, 15, 15, 100, 2, 0))
# Partially Ripe (1): Yellow-Orange
data.extend(generate_samples(200, 150, 60, 10, 12, 10, 100, 2, 1))
# Ripe (2): Deep Orange-Yellow
data.extend(generate_samples(225, 130, 55, 10, 10, 10, 100, 2, 2))
# Decay (3): Brown
data.extend(generate_samples(145, 90, 60, 10, 10, 8, 100, 2, 3))

# ========== ORANGE (fruit_type=3) ==========
# Early Ripe (0): Green-Yellow
data.extend(generate_samples(215, 120, 45, 12, 12, 10, 100, 3, 0))
# Partially Ripe (1): Light Orange
data.extend(generate_samples(232, 130, 35, 10, 12, 8, 100, 3, 1))
# Ripe (2): Bright Orange
data.extend(generate_samples(242, 110, 30, 8, 10, 6, 100, 3, 2))
# Decay (3): Dark Orange-Brown
data.extend(generate_samples(190, 95, 50, 10, 10, 10, 100, 3, 3))

# Create DataFrame
df = pd.DataFrame(data, columns=['R', 'G', 'B', 'fruit_type', 'ripeness_label'])

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df.to_csv('data/fruit_ripeness_dataset.csv', index=False)

print(f"✅ Generated improved dataset: {len(df)} samples")
print(f"\nDistribution:")
for fruit in range(4):
    fruit_name = ['Apple', 'Banana', 'Mango', 'Orange'][fruit]
    print(f"\n{fruit_name}:")
    for ripeness in range(4):
        ripeness_name = ['Early Ripe', 'Partially Ripe', 'Ripe', 'Decay'][ripeness]
        count = len(df[(df['fruit_type'] == fruit) & (df['ripeness_label'] == ripeness)])
        print(f"  {ripeness_name}: {count} samples")
