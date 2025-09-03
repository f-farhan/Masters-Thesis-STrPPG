import os
import numpy as np

def compute_mean_snr(path_to_snr_npy, snr_filename="SNR_MAP_0.npy", snr_threshold=-15):
    """
    Load an SNR heatmap (numpy array), filter out invalid pixels, and compute the mean SNR.

    Parameters:
    - path_to_snr_npy (str): Directory where the .npy file is stored.
    - snr_filename (str): Filename of the SNR map.
    - snr_threshold (float): Threshold below which pixels are considered invalid.

    Returns:
    - mean_snr (float): Mean SNR over valid pixels.
    """
    snr_path = os.path.join(path_to_snr_npy, snr_filename)

    if not os.path.isfile(snr_path):
        raise FileNotFoundError(f"SNR map file not found: {snr_path}")

    snr_map = np.load(snr_path)

    # Filter out invalid or background pixels
    valid_snr = snr_map[~np.isnan(snr_map)]
    valid_snr = valid_snr[valid_snr > snr_threshold]

    mean_snr = np.mean(valid_snr)
    print(f"📊 Mean SNR over valid pixels: {mean_snr:.2f} dB")

    return mean_snr

# Example usage:
if __name__ == "__main__":
    path = "C:/ff/FINAL OUTPUTS/RGB/2/G-R/No Polarization/npy_files"  # Change this to your actual path
    compute_mean_snr(path)
