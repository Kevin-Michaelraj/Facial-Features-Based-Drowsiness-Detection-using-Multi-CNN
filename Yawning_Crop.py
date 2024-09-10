import numpy as np

def extract_timestamps(row):
    timestamps = []
    for i in range(1, len(row), 2):
        start_time = timestamp_to_seconds(str(row[i]))
        end_time = timestamp_to_seconds(str(row[i+1]))
        if not np.isnan(start_time) and not np.isnan(end_time):
            timestamps.append((start_time, end_time))
    return timestamps

def timestamp_to_seconds(timestamp):
    try:
        hours, minutes, seconds = map(float, timestamp.split('.'))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
    except (ValueError, AttributeError):
        return float('nan')


# Path to the Excel file
excel_file_path = 'yawddmale.xlsx'
# Path to the folder containing the original videos
videos_folder = 'yawdd_mirror'
# Path to the folder where cropped videos will be saved
output_folder = 'yawnningmale'

# Read the Excel file
df = pd.read_excel(excel_file_path, header=None)

# Iterate through each row of the Excel file
for index, row in df.iterrows():
    video_name = row[0]
    timestamps = extract_timestamps(row)
    video_path = os.path.join(videos_folder, f'{video_name}.avi')  # Path to the original video (change extension to .avi)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List to hold video segments
    video_segments = []

    # Iterate through each pair of timestamps
    for start_time, end_time in timestamps:
        print(f"Start time: {start_time}, End time: {end_time}")  # Print timestamps for debugging
        
        # Load video clip
        clip = VideoFileClip(video_path).subclip(start_time, end_time)
        video_segments.append(clip)

    # Concatenate video segments
    final_clip = concatenate_videoclips(video_segments)

    # Save the concatenated clip
    output_name = f'{video_name}_cropped.mp4'  # Save as MP4
    output_path = os.path.join(output_folder, output_name)
    final_clip.write_videofile(output_path, codec='libx264', fps=24)  # Specify codec and FPS for MP4

print("Cropping and concatenation completed!")
