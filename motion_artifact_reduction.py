import numpy as np
from scipy.signal import butter, filtfilt

def extract_envelope(signal, fs, cutoff=0.5, order=2):
    nyquist = 0.5 * fs
    norm_cutoff = cutoff / nyquist
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    envelope = filtfilt(b, a, np.abs(signal))
    envelope = np.maximum(envelope, 1e-6)  # Prevent divide-by-zero
    return envelope

def normalize_rppg_tensor(rppg_tensor, fs, cutoff=0.5):
    H, W, T = rppg_tensor.shape
    normalized_tensor = np.zeros_like(rppg_tensor)

    for i in range(H):
        for j in range(W):
            signal = rppg_tensor[i, j, :]
            envelope = extract_envelope(signal, fs, cutoff)
            normalized_tensor[i, j, :] = signal / envelope

    return normalized_tensor
