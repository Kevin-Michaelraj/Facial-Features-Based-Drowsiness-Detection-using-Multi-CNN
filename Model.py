import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, concatenate
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Load image paths and labels from CSV files along with additional features
def load_data_from_csv(csv_file, image_folder):
    df = pd.read_csv(csv_file)
    image_paths = [os.path.join(image_folder, img_name) for img_name in df['Image_Name']]
    labels = [1 if 'yawn' in img_name.lower() else 0 for img_name in df['Image_Name']]
    features = df[['EAR', 'MAR', 'EC', 'MOE', 'LEB', 'SOP']].values
    return image_paths, features, labels

# Load drowsy and vigilant image paths
drowsy_folder = 'yawddpreprocess/drowsy'
vigilant_folder = 'yawddpreprocess/vigilant'
drowsy_csv = 'yawdd-drowsy1.csv'
vigilant_csv = 'yawdd-nondrowsy.csv'

drowsy_image_paths, drowsy_features, drowsy_labels = load_data_from_csv(drowsy_csv, drowsy_folder)
vigilant_image_paths, vigilant_features, vigilant_labels = load_data_from_csv(vigilant_csv, vigilant_folder)

# Combine drowsy and vigilant data
all_image_paths = drowsy_image_paths + vigilant_image_paths
all_features = np.concatenate([drowsy_features, vigilant_features])
all_labels = np.concatenate([drowsy_labels, vigilant_labels])

# Split data into training and validation sets
train_image_paths, val_image_paths, train_features, val_features, train_labels, val_labels = train_test_split(
    all_image_paths, all_features, all_labels, test_size=0.2, random_state=42
)

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if img is None:
        print(f"Could not open or find the image: {image_path}")
        return None
    img = cv2.resize(img, (64, 64))  # Resize to desired input shape
    img = img / 255.0  # Normalize pixel values
    return img

# Print progress indicator for image loading and preprocessing
print("Loading and preprocessing training images:")
train_images = []
with tqdm(total=len(train_image_paths)) as pbar:
    for path in train_image_paths:
        train_images.append(load_and_preprocess_image(path))
        pbar.update(1)
train_images = np.array(train_images)

print("Loading and preprocessing validation images:")
val_images = []
with tqdm(total=len(val_image_paths)) as pbar:
    for path in val_image_paths:
        val_images.append(load_and_preprocess_image(path))
        pbar.update(1)
val_images = np.array(val_images)

# Build CNN model for full images
full_image_input = Input(shape=(64, 64, 1), name='full_image_input')  # Input shape for grayscale images
conv1_full = Conv2D(32, kernel_size=(3, 3), activation='relu')(full_image_input)
maxpool1_full = MaxPooling2D(pool_size=(2, 2))(conv1_full)
conv2_full = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxpool1_full)
maxpool2_full = MaxPooling2D(pool_size=(2, 2))(conv2_full)
flatten_full = Flatten()(maxpool2_full)

# Build CNN model for face bounding boxes
face_bbox_input = Input(shape=(64, 64, 1), name='face_bbox_input')  # Input shape for grayscale images
conv1_face = Conv2D(32, kernel_size=(3, 3), activation='relu')(face_bbox_input)
maxpool1_face = MaxPooling2D(pool_size=(2, 2))(conv1_face)
conv2_face = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxpool1_face)
maxpool2_face = MaxPooling2D(pool_size=(2, 2))(conv2_face)
flatten_face = Flatten()(maxpool2_face)

# CNN branch for additional features
additional_features_input = Input(shape=(6,), name='additional_features_input')  # Additional features
dense1 = Dense(64, activation='relu')(additional_features_input)

# Concatenate the outputs from both branches
concatenated = concatenate([flatten_full, flatten_face, dense1])

# Fully connected layers
dense2 = Dense(128, activation='relu')(concatenated)
dropout = Dropout(0.5)(dense2)
output = Dense(1, activation='sigmoid', name='output')(dropout)

# Define the model with three inputs
model = Model(inputs=[full_image_input, face_bbox_input, additional_features_input], outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([train_images, train_images, train_features], train_labels, epochs=10, batch_size=32, validation_data=([val_images, val_images, val_features], val_labels), verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate([val_images, val_images, val_features], val_labels)
print(f"Validation accuracy: {accuracy * 100:.2f}%")

# Predict labels for validation images
predicted_labels = (model.predict([val_images, val_images, val_features]) > 0.5).astype("int32")

# Calculate confusion matrix
conf_matrix = confusion_matrix(val_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate precision, recall, F1 score, and kappa score
precision = precision_score(val_labels, predicted_labels)
recall = recall_score(val_labels, predicted_labels)
f1 = f1_score(val_labels, predicted_labels)
kappa = cohen_kappa_score(val_labels, predicted_labels)

# Plotting
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Kappa Score']
scores = [accuracy, precision, recall, f1, kappa]
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Kappa Score: {kappa:.2f}")
plt.figure(figsize=(10, 6))
sns.barplot(x=labels, y=scores)
plt.title("Model Evaluation Metrics")
plt.ylabel("Score")
plt.show()

# Save the model
model.save("multi_scale_cnn_with_features_yawdd.h5")
