import cv2
import dlib
import numpy as np
import os
from tqdm import tqdm

def cross_guided_bilateral_filter(target_image, guidance_image, diameter, sigma_color, sigma_space):
    # Apply bilateral filter to the target image using guidance from the guidance image
    filtered_image = cv2.ximgproc.guidedFilter(guidance_image, target_image, diameter, sigma_color, sigma_space)
    return filtered_image

def filter_images_in_folder(input_folder, output_folder, diameter=15, sigma_color=75, sigma_space=75):
    # Ensure output folder exists, create if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize the dlib face detector
    detector = dlib.get_frontal_face_detector()

    # Get list of files in the input folder
    input_files = os.listdir(input_folder)

    # Initialize tqdm progress bar
    progress_bar = tqdm(total=len(input_files), desc='Filtering Images')

    # Iterate over all files in the input folder
    for filename in input_files:
        input_image_path = os.path.join(input_folder, filename)
        
        # Load the input image
        img = cv2.imread(input_image_path)

        # Convert the input image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = detector(gray_img)

        # Create a mask to exclude face regions
        mask = np.ones_like(gray_img, dtype=np.uint8)
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)  # Set face regions to 0

        # Apply bilateral filter to the whole region
        non_face_region = cross_guided_bilateral_filter(gray_img, gray_img, diameter, sigma_color, sigma_space)

        # Apply the mask to the input image to get the non-face region
        face_region = cv2.bitwise_and(gray_img, gray_img, mask=cv2.bitwise_not(mask))

        # Apply bilateral filter to the non-face region
        filtered_face_region = cross_guided_bilateral_filter(face_region, face_region, diameter, sigma_color, sigma_space)

        # Combine the filtered face and non-face regions
        filtered_result = cv2.bitwise_or(filtered_face_region, non_face_region)

        # Save the filtered result
        output_image_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_image_path, filtered_result)

        # Update progress bar
        progress_bar.update(1)

    # Close progress bar
    progress_bar.close()
    print("Filtering completed.")

# Define input and output folders
input_folder = "yawddkeyframe/vigilant"
output_folder = "yawddpreprocess/vigilant"

# Define filter parameters
diameter = 15
sigma_color = 75
sigma_space = 75

# Apply filter to images in input folder and save in output folder
filter_images_in_folder(input_folder, output_folder, diameter, sigma_color, sigma_space)
