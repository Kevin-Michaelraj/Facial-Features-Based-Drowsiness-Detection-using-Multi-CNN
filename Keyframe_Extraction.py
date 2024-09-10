import cv2
import os
import dlib

def extract_keyframes(video_path, output_path, interval=10):
    """
    Extract keyframes from a video at a specified interval if a face is detected.

    Args:
    - video_path (str): Path to the input video file.
    - output_path (str): Directory to save the extracted keyframes.
    - interval (int): Interval between keyframes (in frames).

    Returns:
    - None
    """
    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % interval == 0:
            # Detect faces in the frame
            faces = detector(frame)

            if faces:
                # Save the keyframe
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                keyframe_path = os.path.join(output_path, f"{video_name}_keyframe_{frame_count}.jpg")
                cv2.imwrite(keyframe_path, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Keyframes extraction completed for {video_path}")

def extract_keyframes_from_folder(folder_path, output_path, interval=30):
    """
    Extract keyframes from all video files in a folder.

    Args:
    - folder_path (str): Path to the folder containing video files.
    - output_path (str): Directory to save the extracted keyframes.
    - interval (int): Interval between keyframes (in frames).

    Returns:
    - None
    """

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        # Check if the file is a video
        if os.path.isfile(filepath) and filename.lower().endswith(('.avi', '.mp4', '.mov')):
            # Extract keyframes from the video
            extract_keyframes(filepath, output_path, interval)

# For Yawdd-dashboard-male
videos_folder = "yawdd/vigilant"
output_folder = "yawddkeyframe/vigilant" 
extract_keyframes_from_folder(videos_folder, output_folder, interval=10)
