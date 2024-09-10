import os
import cv2
import dlib
import numpy as np
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bar

# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to calculate distance between two points
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye_points):
    return (distance(eye_points[1], eye_points[5]) + distance(eye_points[2], eye_points[4])) / (2.0 * distance(eye_points[0], eye_points[3]))

# Function to calculate Mouth Aspect Ratio (MAR)
def calculate_mar(mouth_points):
    return distance(mouth_points[14], mouth_points[18]) / distance(mouth_points[12], mouth_points[16])

# Function to calculate Eye Circularity (EC)
def calculate_ec(eye_points):
    pupil_area = (distance(eye_points[1], eye_points[4]) / 2) ** 2 * np.pi
    eye_perimeter = sum([distance(eye_points[i], eye_points[(i + 1) % 6]) for i in range(6)])
    return (4 * np.pi * pupil_area) / (eye_perimeter ** 2)

# Function to calculate Mouth Over Eye (MOE)
def calculate_moe(ear, mar):
    return mar / ear

# Function to calculate Level of Eyebrows (LEB)
def calculate_leb(eyebrow_points, eye_point):
    return (distance(eyebrow_points[0], eye_point) + distance(eyebrow_points[1], eye_point)) / 2

# Function to calculate Size of Pupils (SOP)
def calculate_sop(eye_points):
    return distance(eye_points[1], eye_points[4]) / distance(eye_points[0], eye_points[3])

# Directory containing images
folder_path = 'yawddpreprocess/vigilant'

# List to store all features
all_features = []

# Get total number of image files in the directory
total_files = len([filename for filename in os.listdir(folder_path)
                   if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')])

# Initialize tqdm with total number of iterations
with tqdm(total=total_files) as pbar:
    # Iterate over each image file in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Load the image
            image = cv2.imread(os.path.join(folder_path, filename))

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = detector(gray)

            # Iterate over each face in the image
            for face in faces:
                landmarks = predictor(gray, face)
                landmarks_points = [(p.x, p.y) for p in landmarks.parts()]

                # Calculate facial features
                left_eye_ear = calculate_ear(landmarks_points[36:42])
                right_eye_ear = calculate_ear(landmarks_points[42:48])
                ear = (left_eye_ear + right_eye_ear) / 2.0
                mar = calculate_mar(landmarks_points[48:68])
                left_eye_ec = calculate_ec(landmarks_points[36:42])
                right_eye_ec = calculate_ec(landmarks_points[42:48])
                ec = (left_eye_ec + right_eye_ec) / 2.0
                moe = calculate_moe(ear, mar)
                left_leb = calculate_leb(landmarks_points[17:22], landmarks_points[36])
                right_leb = calculate_leb(landmarks_points[22:27], landmarks_points[45])
                leb = (left_leb + right_leb) / 2.0
                left_eye_sop = calculate_sop(landmarks_points[36:42])
                right_eye_sop = calculate_sop(landmarks_points[42:48])
                sop = (left_eye_sop + right_eye_sop) / 2.0

                # Append features to the list along with image name
                all_features.append([filename, ear, mar, ec, moe, leb, sop])
            
            # Update progress bar after processing each image
            pbar.update(1)

# Create a DataFrame from the list of features
df = pd.DataFrame(all_features, columns=['Image_Name', 'EAR', 'MAR', 'EC', 'MOE', 'LEB', 'SOP'])

# Save DataFrame to CSV file
df.to_csv('yawdd-nondrowsy.csv', index=False)

print("Features saved to nthuddd-nondrowsy.csv")
