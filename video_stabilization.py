import os
import cv2
import numpy as np
import tempfile
import ffmpeg
import matplotlib.pyplot as plt
from vidstab import VidStab


def compute_motion_magnitude(video_path, output_dir, fps=50, section_duration=10, max_duration=60):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Cannot open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or fps
    video_duration = total_frames / frame_rate

    if video_duration > max_duration:
        video_duration = max_duration

    frames_per_section = int(section_duration * frame_rate)
    total_frames = int(video_duration * frame_rate)
    num_sections = int(video_duration // section_duration)

    motion_magnitudes = np.zeros(num_sections)

    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Error: Cannot read first frame.")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_idx, section_idx = 0, 0

    while frame_idx < total_frames:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        motion_magnitude = np.linalg.norm(flow)

        motion_magnitudes[section_idx] += motion_magnitude

        prev_gray = curr_gray
        frame_idx += 1
        if frame_idx % frames_per_section == 0:
            section_idx += 1
            if section_idx >= num_sections:
                break

    cap.release()

    best_section = np.argmin(motion_magnitudes)
    best_start_time = best_section * section_duration

    # Plot motion magnitudes
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(np.arange(1, num_sections + 1), motion_magnitudes, marker='o', color='blue', label="Motion Magnitude")
    plt.axvline(x=best_section + 1, color='red', linestyle='--', label=f"Least Motion (Section {best_section + 1})")
    plt.xlabel("Section Number")
    plt.ylabel("Motion Magnitude")
    plt.title("Motion Analysis Across Video Sections")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "motion_analysis.png"))
    plt.close()

    return best_start_time, section_duration, frame_rate


def stabilize_selected_section_return_binary(video_path, start_time, duration, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    temp_trim_path = os.path.join(output_dir, "temp_trimmed.avi")
    stabilized_path = os.path.join(output_dir, "stabilized.avi")

    # Trim using ffmpeg
    ffmpeg.input(video_path, ss=start_time, t=duration).output(temp_trim_path, codec="copy").run()

    # Stabilize
    stabilizer = VidStab()
    stabilizer.stabilize(
        input_path=temp_trim_path,
        output_path=stabilized_path,
        border_type="black",
        playback=False,
        smoothing_window=15,
        output_fourcc="FFV1"
    )

    # Save trajectory plot
    try:
        stabilizer.plot_trajectory()
        plt.title("Stabilization Trajectory")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "stabilization_trajectory.png"))
        plt.close()
    except Exception as e:
        print(f"⚠️ Could not generate trajectory plot: {e}")

    # Save transform plot
    try:
        stabilizer.plot_transforms()
        plt.title("Applied Stabilization Transforms")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "stabilization_transforms.png"))
        plt.close()
    except Exception as e:
        print(f"⚠️ Could not generate transforms plot: {e}")

    # Return binary video
    with open(stabilized_path, "rb") as stab_video:
        stab_binary = stab_video.read()

    return stab_binary



def auto_stabilize_return_video(video_path, section_duration=10, max_duration=60, output_dir="stabilization_outputs"):
    start_time, duration, _ = compute_motion_magnitude(
        video_path,
        output_dir=output_dir,
        section_duration=section_duration,
        max_duration=max_duration
    )
    stabilized_video_binary = stabilize_selected_section_return_binary(
        video_path,
        start_time,
        duration,
        output_dir=output_dir
    )
    return stabilized_video_binary


# ✅ Example usage
if __name__ == "__main__":
    input_file = "D:/FF/Monochrome Acquisition/orange_1.avi"
    output_folder = "C:/ff/outputs/tests/test"
    video_data = auto_stabilize_return_video(input_file, output_dir=output_folder)
    print(f"✅ Stabilized video binary returned ({len(video_data)} bytes).")
