import os
import pandas as pd
import matplotlib.pyplot as plt

# ✅ Set your dataset path (make sure this is correct)
base_path = r"C:\Users\Puja\Downloads\deep learning project"

# ✅ Function to load ECG file (without assuming headers)
def load_ecg(subject, activity):
    filepath = os.path.join(base_path, subject, activity, 'ECG.tsv')
    return pd.read_csv(filepath, sep='\t', header=None)  # no header

# ✅ Load and preview the file
df = load_ecg('subject_00', 'sitting')
print(df.head())
print(df.shape)

import os
import pandas as pd
import matplotlib.pyplot as plt

base_path = r"C:\Users\Puja\Downloads\deep learning project"

def load_ecg(subject, activity):
    filepath = os.path.join(base_path, subject, activity, 'ECG.tsv')
    return pd.read_csv(filepath, delim_whitespace=True, header=None)

df = load_ecg('subject_00', 'sitting')
print(df.head())
print(df.shape)

plt.figure(figsize=(12, 4))
plt.plot(df[0], label='ECG Signal')  # Try df[1] if this looks flat
plt.title('Raw ECG - subject_00 - sitting')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

pip install neurokit2

import neurokit2 as nk

# 1. Extract the ECG signal from column 0 (or whichever is correct)
ecg_signal = df[0].values

# 2. Preprocess and detect R-peaks
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=250)
ecg_signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=250)

# 3. Plot ECG with detected R-peaks
nk.ecg_plot(ecg_signals, info)

import numpy as np
from scipy.stats import skew, kurtosis

# Use rpeaks from previous Step 2
rpeaks = info["ECG_R_Peaks"]

# Convert to R-R intervals (ms) from indices
rr_intervals = np.diff(rpeaks) * (1000 / 250)  # 250 Hz sampling

# Time-domain features
AHR = 60000 / np.mean(rr_intervals)
SDNN = np.std(rr_intervals)
RMSSD = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
NN50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)

# Statistical features
stress_index = len(rr_intervals) / np.std(rr_intervals)
skewness_val = skew(rr_intervals)
kurtosis_val = kurtosis(rr_intervals)
vse = np.var(rr_intervals) / np.mean(rr_intervals)

# Approximate frequency domain (for educational use)
lf_power = np.sum((rr_intervals > 100) & (rr_intervals < 300))  # fake LF proxy
hf_power = np.sum((rr_intervals > 300) & (rr_intervals < 600))  # fake HF proxy
lf_hf_ratio = lf_power / hf_power if hf_power != 0 else 0

# Final dictionary
features = {
    "AHR": AHR,
    "SDNN": SDNN,
    "RMSSD": RMSSD,
    "NN50": NN50,
    "Stress_Index": stress_index,
    "Skewness": skewness_val,
    "Kurtosis": kurtosis_val,
    "VSE": vse,
    "SNS_index": lf_power,
    "PNS_index": hf_power,
    "SNS_to_PNS_ratio": lf_hf_ratio
}

# Show results
for name, value in features.items():
    print(f"{name}: {value:.2f}")

import os

# ✅ Step 3A: Define base path and dataset structure
base_path = r"C:\Users\Puja\Downloads\deep learning project"
subjects = [f"subject_0{i}" for i in range(5)]  # subject_00 to subject_04
activities = ['sitting', 'maths', 'walking', 'jogging', 'hand_bike']

import numpy as np
import neurokit2 as nk
from scipy.stats import skew, kurtosis
import pandas as pd

# ✅ Step 3B: Extract features from an ECG signal
def extract_features(ecg_signal, sampling_rate=250):
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    _, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
    rpeaks = info["ECG_R_Peaks"]

    rr_intervals = np.diff(rpeaks) * (1000 / sampling_rate)

    AHR = 60000 / np.mean(rr_intervals)
    SDNN = np.std(rr_intervals)
    RMSSD = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    NN50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
    stress_index = len(rr_intervals) / np.std(rr_intervals)
    skewness_val = skew(rr_intervals)
    kurtosis_val = kurtosis(rr_intervals)
    vse = np.var(rr_intervals) / np.mean(rr_intervals)
    lf_power = np.sum((rr_intervals > 100) & (rr_intervals < 300))
    hf_power = np.sum((rr_intervals > 300) & (rr_intervals < 600))
    lf_hf_ratio = lf_power / hf_power if hf_power != 0 else 0

    return {
        "AHR": AHR,
        "SDNN": SDNN,
        "RMSSD": RMSSD,
        "NN50": NN50,
        "Stress_Index": stress_index,
        "Skewness": skewness_val,
        "Kurtosis": kurtosis_val,
        "VSE": vse,
        "SNS_index": lf_power,
        "PNS_index": hf_power,
        "SNS_to_PNS_ratio": lf_hf_ratio
    }

import numpy as np
import neurokit2 as nk
from scipy.stats import skew, kurtosis

def extract_features(ecg_signal, sampling_rate=250):
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    _, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
    rpeaks = info["ECG_R_Peaks"]

    rr_intervals = np.diff(rpeaks) * (1000 / sampling_rate)

    AHR = 60000 / np.mean(rr_intervals)
    SDNN = np.std(rr_intervals)
    RMSSD = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    NN50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
    stress_index = len(rr_intervals) / np.std(rr_intervals)
    skewness_val = skew(rr_intervals)
    kurtosis_val = kurtosis(rr_intervals)
    vse = np.var(rr_intervals) / np.mean(rr_intervals)
    lf_power = np.sum((rr_intervals > 100) & (rr_intervals < 300))
    hf_power = np.sum((rr_intervals > 300) & (rr_intervals < 600))
    lf_hf_ratio = lf_power / hf_power if hf_power != 0 else 0

    return {
        "AHR": AHR,
        "SDNN": SDNN,
        "RMSSD": RMSSD,
        "NN50": NN50,
        "Stress_Index": stress_index,
        "Skewness": skewness_val,
        "Kurtosis": kurtosis_val,
        "VSE": vse,
        "SNS_index": lf_power,
        "PNS_index": hf_power,
        "SNS_to_PNS_ratio": lf_hf_ratio
    }

if os.path.exists(file_path):
    try:
        # ✅ Load ECG data with updated separator
        df = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python')

        # Assume ECG is in column 0
        ecg_signal = df.iloc[:, 0].values

        # ✅ Call the extract_features function (make sure it's defined)
        features = extract_features(ecg_signal, sampling_rate=250)
        features["Subject"] = subject
        features["Activity"] = activity

        all_features.append(features)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
else:
    print(f"🚫 File not found: {file_path}")

# Assuming all_features is already populated
features_df = pd.DataFrame(all_features)

for i in range(min(5, len(all_features))):
    print(all_features[i])

subjects = [f"subject_{i:02d}" for i in range(25)]  # subjects 00 to 24
activities = ['sitting', 'maths', 'walking', 'jogging', 'hand_bike']

all_features = []

for subject in subjects:
    for activity in activities:
        file_path = os.path.join(base_path, subject, activity, 'ECG.tsv')
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python')
                ecg_signal = df.iloc[:, 0].values
                features = extract_features(ecg_signal, sampling_rate=250)
                features["Subject"] = subject
                features["Activity"] = activity
                all_features.append(features)
                print(f"✅ Added features for {subject} - {activity}")
            except Exception as e:
                print(f"❌ Error processing {subject} - {activity}: {e}")
        else:
            print(f"🚫 File not found: {file_path}")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Convert list of feature dictionaries to DataFrame
features_df = pd.DataFrame(all_features)

# Check if features were extracted
print(f"Total entries: {len(features_df)}")
print("Available activities:", features_df['Activity'].unique())

# Drop non-feature columns to get X
X = features_df.drop(columns=['Subject', 'Activity'])

# Target variable
y = features_df['Activity']

# Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Predict on training data (or split into train/test if needed)
y_pred = model.predict(X)

# Evaluation
print("\n📊 Classification Report:")
print(classification_report(y, y_pred))

# Confusion Matrix
cm = confusion_matrix(y, y_pred, labels=sorted(y.unique()))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()), cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)

# Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Get predictions
y_pred = clf.predict(X_test)

# Classes
classes = clf.classes_

# Plot confusion matrix for each class
for i, label in enumerate(classes):
    # Create one-vs-all ground truth and prediction
    y_true_bin = (y_test == label).astype(int)
    y_pred_bin = (y_pred == label).astype(int)

    cm = confusion_matrix(y_true_bin, y_pred_bin)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not " + label, label])

    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title(f"Confusion Matrix: {label}")
    plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import butter, filtfilt

# ====================
# SETUP
# ====================
activities = ['hand_bike', 'jogging', 'sitting', 'walking', 'maths']
output_dir = 'scalograms'
os.makedirs(output_dir, exist_ok=True)

# ====================
# Synthetic ECG signal generator for each class
# ====================
def generate_dummy_ecg(activity, duration_sec=5, fs=250):
    t = np.linspace(0, duration_sec, duration_sec * fs)

    if activity == 'hand_bike':
        signal = np.sin(2 * np.pi * 1.0 * t) + 0.3 * np.random.randn(len(t))
    elif activity == 'jogging':
        signal = np.sin(2 * np.pi * 2.0 * t) + 0.3 * np.random.randn(len(t))
    elif activity == 'sitting':
        signal = np.sin(2 * np.pi * 0.5 * t) + 0.3 * np.random.randn(len(t))
    elif activity == 'walking':
        signal = np.sin(2 * np.pi * 1.5 * t) + 0.3 * np.random.randn(len(t))
    elif activity == 'maths':
        signal = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.random.randn(len(t))
    else:
        signal = np.sin(2 * np.pi * 1.7 * t) + 0.5 * np.random.randn(len(t))

    return signal

# ====================
# Bandpass filter
# ====================
def bandpass_filter(signal, fs=250, low=0.5, high=40, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

# ====================
# Save scalogram
# ====================
def save_scalogram(signal, filename, fs=250):
    widths = np.arange(1, 128)
    coef, _ = pywt.cwt(signal, widths, 'cmor1.5-1.0', sampling_period=1/fs)

    plt.figure(figsize=(2.27, 2.27), dpi=100)  # 227x227 image
    plt.imshow(np.abs(coef), extent=[0, len(signal)/fs, 1, 128], cmap='jet', aspect='auto')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

# ====================
# GENERATE IMAGES
# ====================
num_images_per_class = 600  # total 600 * 5 = 3000

for cls in activities:
    class_dir = os.path.join(output_dir, cls)
    os.makedirs(class_dir, exist_ok=True)

    for i in range(num_images_per_class):
        raw = generate_dummy_ecg(cls)  # ✅ Class-specific signal generation
        filtered = bandpass_filter(raw)
        filepath = os.path.join(class_dir, f"{cls}_{i:04d}.png")
        save_scalogram(filtered, filepath)
        print(f"Saved: {filepath}")

import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Image parameters
IMG_SIZE = (227, 227)
BATCH_SIZE = 32

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    'scalograms',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='categorical'
)

val_gen = train_datagen.flow_from_directory(
    'scalograms',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='validation',
    class_mode='categorical'
)

# MobileNetV2 model
base_model = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=10)

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=5)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Get true labels from the validation generator
val_gen.reset()  # Important if you previously ran predictions
Y_true = val_gen.classes

# Get class labels
class_labels = list(val_gen.class_indices.keys())

# Predict
Y_pred_prob = model.predict(val_gen, verbose=1)
Y_pred = np.argmax(Y_pred_prob, axis=1)

# Classification Report
print("Classification Report:")
print(classification_report(Y_true, Y_pred, target_names=class_labels))

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(Y_true, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

import numpy as np
from sklearn.metrics import confusion_matrix

labels = ['hand_bike', 'jogging', 'maths', 'sitting', 'walking']
for i, label in enumerate(labels):
    y_true_binary = (y_true == i).astype(int)
    y_pred_binary = (y_pred == i).astype(int)
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    print(f"Confusion matrix for class '{label}':\n{cm}\n")
