import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model

# Load the detector and predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

# Load the trained model
model = load_model("multi_scale_cnn_with_features_yawdd.h5")

# Function to preprocess image for prediction
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not open or find the image: {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) > 0:
        face = faces[0]  # Assuming there is only one face
        landmarks = predictor(gray, face)
        landmarks_points = [(p.x, p.y) for p in landmarks.parts()]
        ear = calculate_ear(landmarks_points[36:42])
        mar = calculate_mar(landmarks_points[48:68])
        ec = calculate_ec(landmarks_points[36:42])
        moe = calculate_moe(ear, mar)
        leb = calculate_leb(landmarks_points[17:22], landmarks_points[36])
        sop = calculate_sop(landmarks_points[36:42])
        features = np.array([[ear, mar, ec, moe, leb, sop]])
        return features
    else:
        print("No faces detected in the image.")
        return None

# Path to the image for prediction
image_path = "yawddkeyframe/vigilant/1-FemaleNoGlasses-Normal_keyframe_0.jpg"

# Preprocess the image
features = preprocess_image(image_path)
if features is not None:
    # Load the image
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image_gray = cv2.resize(image_gray, (64, 64))  # Resize to desired input shape
    image_gray = image_gray / 255.0  # Normalize pixel values

    # Expand dimensions to match model input shape
    image_gray = np.expand_dims(image_gray, axis=0)

    # Make prediction using the model
    prediction = model.predict([image_gray, image_gray, features])
    if prediction > 0.5:
        print("The person in the image is drowsy.")
    else:
        print("The person in the image is not drowsy.")
