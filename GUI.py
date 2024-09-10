import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import dlib

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

# Function to preprocess image for prediction
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", f"Could not open or find the image: {image_path}")
        return None, None

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

        # Preprocess the image for prediction
        image_gray = cv2.resize(gray, (64, 64))  # Resize to desired input shape
        image_gray = image_gray / 255.0  # Normalize pixel values
        image_gray = np.expand_dims(image_gray, axis=0)  # Expand dimensions to match model input shape

        return image_gray, features
    else:
        messagebox.showerror("Error", "No faces detected in the image.")
        return None, None

# Function to make prediction using the model
def predict_drowsiness(image_path, selected_model):
    if selected_model == "yawdd":
        model_path = "multi_scale_cnn_with_features_yawdd.h5"
    elif selected_model == "nthuddd":
        model_path = "multi_scale_cnn_with_features.h5"
    else:
        messagebox.showerror("Error", "Invalid model selection.")
        return

    model = load_model(model_path)

    image_gray, features = preprocess_image(image_path)
    if image_gray is not None and features is not None:
        # Make prediction using the model
        prediction = model.predict([image_gray, image_gray, features])
        if prediction > 0.5:
            messagebox.showinfo("Prediction", "The person in the image is drowsy.")
        else:
            messagebox.showinfo("Prediction", "The person in the image is not drowsy.")

# Function to handle button click for image selection and prediction
def select_image_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        selected_model = model_choice.get()
        predict_drowsiness(file_path, selected_model)

# Create the main application window
root = tk.Tk()
root.title("Drowsiness Detection")

# Add background color to the window
root.configure(bg="#f0f0f0")

# Create a frame for the model selection
model_frame = tk.Frame(root, bg="#f0f0f0", pady=10)
model_frame.pack()

# Add a label for model selection
model_label = tk.Label(model_frame, text="Select Model:", bg="#f0f0f0", font=("Helvetica", 12))
model_label.pack(side="left")

# Add a dropdown for model selection
model_choice = tk.StringVar(root)
model_choice.set("yawdd")  # Default selection
model_dropdown = tk.OptionMenu(model_frame, model_choice, "yawdd", "nthuddd")
model_dropdown.pack(side="left")

# Create a button to select an image
select_image_button = tk.Button(root, text="Select Image", command=select_image_and_predict)
select_image_button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
