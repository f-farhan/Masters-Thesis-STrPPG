import numpy as np
from scipy.signal import find_peaks
from scipy.stats import pearsonr

def compute_template_correlation_sqi(signal, fs, plot=False):
    """
    Compute the Signal Quality Index (SQI) based on template correlation.
    The function segments the signal into pulses, computes a template pulse,
    and evaluates the average correlation of all pulses with the template.

    Parameters:
    - signal: 1D numpy array of the rPPG signal.
    - fs: Sampling frequency (Hz).
    - plot: If True, plots pulses and template (for debugging/visualization).

    Returns:
    - mean_correlation: The average correlation coefficient (SQI value).
    """
    peaks, _ = find_peaks(signal, distance=fs*0.4)  # ~150 bpm max → 0.4s min distance
    pulse_segments = []
    window = int(0.6 * fs)  # 0.6 seconds per pulse segment

    for peak in peaks:
        start = peak - window // 2
        end = peak + window // 2
        if start >= 0 and end <= len(signal):
            segment = signal[start:end]
            segment = (segment - np.mean(segment)) / np.std(segment)  # Normalize
            pulse_segments.append(segment)

    if len(pulse_segments) < 3:
        return 0.0  # Not enough segments to compute meaningful SQI

    pulses = np.stack(pulse_segments)
    template = np.mean(pulses, axis=0)
    correlations = [pearsonr(pulse, template)[0] for pulse in pulses]
    mean_correlation = np.mean(correlations)

    return mean_correlation
