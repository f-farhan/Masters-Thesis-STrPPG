import cv2
import os

def save_first_frame(video_path, output_dir, output_filename='roi_ref_visual.png'):
    """
    Save the first frame of a video as a PNG image.

    Parameters:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where the PNG will be saved.
        output_filename (str): Name of the output image file (default is 'roi_ref_visual.png').

    Returns:
        str: Full path to the saved image, or None if failed.
    """
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read the first frame.")
        return None

    cv2.imwrite(output_path, frame)
    print(f"✅ First frame saved as {output_path}")
    return output_path
