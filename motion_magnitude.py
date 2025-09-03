## This code calculates the average motion magnitude of two videos ##
## Can be used to compare the magnitude of motion before and after stabilization ##

import cv2
import numpy as np

def average_motion_magnitude(video_path, max_duration=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Error: Cannot open video file: {video_path}")

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_duration is not None:
        total_frames = min(total_frames, int(frame_rate * max_duration))

    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("❌ Error: Cannot read the first frame.")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    total_motion_magnitude = 0.0
    frame_count = 0

    for _ in range(1, total_frames):
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        motion_magnitude = np.linalg.norm(flow)
        total_motion_magnitude += motion_magnitude
        frame_count += 1

        prev_gray = curr_gray

    cap.release()

    average_magnitude = total_motion_magnitude / frame_count if frame_count else 0.0
    return average_magnitude

# Example Usage:
video_path1 = "C:/ff/FINAL OUTPUTS/Motion Compensation/4/temp_trimmed.avi"
video_path2 = "C:/ff/FINAL OUTPUTS/Motion Compensation/4/stabilized.avi"

avg1 = average_motion_magnitude(video_path1)
avg2 = average_motion_magnitude(video_path2)

print(f"📊 Average Motion Magnitude for trimmed video: {avg1:.2f}")
print(f"📊 Average Motion Magnitude for stabilized video: {avg2:.2f}")
