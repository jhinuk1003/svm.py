# main.py (Fixed with class_weight and safe image check)
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import os

# Function to extract HSV features from an image
def extract_average_hsv(image_path, grid_shape=(4, 5)):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width, _ = hsv.shape
    cell_h, cell_w = height // grid_shape[0], width // grid_shape[1]

    features = []
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            cell = hsv[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            mean_hsv = np.mean(cell.reshape(-1, 3), axis=0)
            features.append(mean_hsv)
    return np.array(features)

# Load cleaned CSV
try:
    df = pd.read_csv("notebooks/well_data_clean.csv")
except Exception as e:
    print("❌ Error reading CSV:", e)
    exit()

# Drop rows with NaN in Label column
df = df.dropna(subset=['Label'])

# Convert Label to integer
try:
    df['Label'] = df['Label'].astype(int)
except Exception as e:
    print("❌ Error converting Label to int:", e)
    print(df['Label'].unique())
    exit()

# Show class balance
print("\n📊 Label Distribution:")
print(df['Label'].value_counts())

# Prepare features and labels
X = df[['H', 'S', 'V']].values
y = df['Label'].values

# Check for any NaNs
if pd.isnull(y).any():
    print("❌ y still contains NaN values. Aborting.")
    exit()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM with class_weight='balanced'
clf = SVC(kernel='rbf', gamma='scale', class_weight='balanced')
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Predict unknown sample image (optional)
unknown_image_path = "data/sample 1.png"
if os.path.exists(unknown_image_path):
    try:
        unknown_features = extract_average_hsv(unknown_image_path)
        unknown_predictions = clf.predict(unknown_features)
        print("\n🔎 Predictions for Unknown Sample:")
        for i, pred in enumerate(unknown_predictions):
            print(f"Well {i+1}: {'Positive' if pred == 1 else 'Negative'}")
    except Exception as e:
        print("⚠️ Unable to process unknown image:", e)
else:
    print(f"⚠️ File '{unknown_image_path}' not found. Skipping prediction.")
