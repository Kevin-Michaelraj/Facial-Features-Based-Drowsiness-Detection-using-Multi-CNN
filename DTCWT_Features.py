import cv2
import dlib
import numpy as np
from dtcwt.numpy import Transform2d
import os
import csv

# Function to detect facial landmarks using dlib
def detect_facial_landmarks(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks_array = []
    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_array.append([x, y])
    return np.array(landmarks_array)

# Function to perform Dual Tree Complex Wavelet Transform (DTCWT) feature extraction
def dtcwt_feature_extraction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    transform = Transform2d()
    coeffs = transform.forward(gray, nlevels=3)
    energy_features = [np.sum(np.abs(coeffs.highpasses[i])**2) for i in range(len(coeffs.highpasses))]
    return energy_features

# Define the Walsh-Hadamard Transform function
def apply_wht(input_data):
    input_data = np.array(input_data)
    if len(input_data) == 1:
        return input_data
    half_length = len(input_data) // 2
    lower_half = input_data[:half_length]
    upper_half = input_data[half_length:]
    even_sum = lower_half + upper_half
    odd_diff = lower_half - upper_half
    lower_transformed = apply_wht(even_sum)
    upper_transformed = apply_wht(odd_diff)
    return np.concatenate((lower_transformed, upper_transformed))

# Function to save features to a CSV file
def save_features_to_csv(features, image_name, csv_filename):
    str_features = [str(feature) for feature in features]
    with open(csv_filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([str(image_name)] + str_features)

# Function to process a single image
def process_image(image_path):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image: {image_path}")
        return
    landmarks = detect_facial_landmarks(image)
    if landmarks is None:
        print(f"No faces detected in the image: {image_name}")
        return

    try:
        if len(landmarks) >= 37:
            eye_left_x, eye_left_y = landmarks[36]
            patch_size = 32
            eye_left_patch = image[eye_left_y - patch_size // 2: eye_left_y + patch_size // 2,
                                   eye_left_x - patch_size // 2: eye_left_x + patch_size // 2]
            dtcwt_features_eye_left = dtcwt_feature_extraction(eye_left_patch)
            wht_features_eye_left = apply_wht(dtcwt_features_eye_left)
            if len(landmarks) >= 46:
                eye_right_x, eye_right_y = landmarks[45]
                eye_right_patch = image[eye_right_y - patch_size // 2: eye_right_y + patch_size // 2,
                                        eye_right_x - patch_size // 2: eye_right_x + patch_size // 2]
                dtcwt_features_eye_right = dtcwt_feature_extraction(eye_right_patch)
                wht_features_eye_right = apply_wht(dtcwt_features_eye_right)
                if len(landmarks) >= 55:
                    mouth_left_x, mouth_left_y = landmarks[48]
                    mouth_right_x, mouth_right_y = landmarks[54]
                    mouth_patch = image[mouth_left_y: mouth_right_y, mouth_left_x: mouth_right_x]
                    dtcwt_features_mouth = dtcwt_feature_extraction(mouth_patch)
                    wht_features_mouth = apply_wht(dtcwt_features_mouth)

                    max_length = max(len(dtcwt_features_eye_left), len(wht_features_eye_left),
                                     len(dtcwt_features_eye_right), len(wht_features_eye_right),
                                     len(dtcwt_features_mouth), len(wht_features_mouth))
                    dtcwt_features_eye_left = np.pad(dtcwt_features_eye_left, (0, max_length - len(dtcwt_features_eye_left)))
                    wht_features_eye_left = np.pad(wht_features_eye_left, (0, max_length - len(wht_features_eye_left)))
                    dtcwt_features_eye_right = np.pad(dtcwt_features_eye_right, (0, max_length - len(dtcwt_features_eye_right)))
                    wht_features_eye_right = np.pad(wht_features_eye_right, (0, max_length - len(wht_features_eye_right)))
                    dtcwt_features_mouth = np.pad(dtcwt_features_mouth, (0, max_length - len(dtcwt_features_mouth)))
                    wht_features_mouth = np.pad(wht_features_mouth, (0, max_length - len(wht_features_mouth)))

                    save_features_to_csv([*dtcwt_features_eye_left ,*wht_features_eye_left ,*dtcwt_features_eye_right ,*wht_features_eye_right ,*dtcwt_features_mouth ,*wht_features_mouth], image_name, 'dash_male.csv')
                    print(f"DTCWT and WHT features saved for image {image_name}.")
                else:
                    print(f"Error: Mouth landmarks not detected for image {image_name}.")
            else:
                print(f"Error: Right eye landmarks not detected for image {image_name}.")
        else:
            print(f"Error: Left eye landmarks not detected for image {image_name}.")
    except Exception as e:
        print(f"Error processing image {image_name}")

# Function to process images in a folder
def process_images_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
    for image_file in image_files:
        process_image(image_file)

# Example usage:
images_folder = 'preprocessed/dash_male'
process_images_in_folder(images_folder)
