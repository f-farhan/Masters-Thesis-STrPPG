#=============== Import Libraries ===============#

import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from joblib import Parallel, delayed
from scipy.signal import find_peaks
from config import cf

def plot_fft_with_band(signal, timeTrace, lowF, upF, h2_lowF, h2_upF, pixel_coords=None, save_path=None):
    n = len(signal)
    dt = timeTrace[1] - timeTrace[0]
    freqs = np.fft.fftfreq(n, d=dt)
    window = np.hanning(n)
    fft_vals = np.fft.fft(signal * window)

    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_power_db = 20 * np.log10(np.abs(fft_vals[pos_mask]) + 1e-10)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, fft_power_db, label="FFT Magnitude (dB)", linewidth=1.5)
    plt.axvspan(lowF, upF, color='green', alpha=0.3, label='1st Harmonic')
    plt.axvspan(h2_lowF, h2_upF, color='orange', alpha=0.3, label='2nd Harmonic')

    centre_freq = (lowF + upF) / 2
    plt.axvline(x=centre_freq, color='red', linestyle='--', linewidth=2, label=f'Center: {centre_freq:.2f} Hz')
    plt.text(centre_freq + 0.05, max(fft_power_db) - 5, f'{centre_freq:.2f} Hz', color='red', fontsize=9)

    title = "FFT Spectrum of rPPG Signal"
    if pixel_coords is not None:
        title += f" (Pixel {pixel_coords})"
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()
    plt.close()

def snr_heatmap(res_path, pulseTraces, timeTrace, refTraces, num_heatmaps, roi_coords=None):
    if cf.DEBUG == 1:
        pulseTraces = np.load(os.path.join(res_path, "npy_files/pulseTraces.npy"))
        timeTrace = np.load(os.path.join(res_path, "npy_files/timeTrace.npy"))
        refTraces = np.load(os.path.join(res_path, "npy_files/rppg_ref.npy"))

    total_time = np.max(timeTrace) - np.min(timeTrace)
    time_per_heatmap = total_time / num_heatmaps

    for i in range(num_heatmaps):
        start_time = i * time_per_heatmap
        end_time = start_time + time_per_heatmap

        mask = (timeTrace >= start_time) & (timeTrace < end_time)
        current_pulseTraces = pulseTraces[:, :, mask]
        current_timeTrace = timeTrace[mask]
        current_refTraces = refTraces[mask]

        n = len(current_refTraces)
        window = np.hanning(n)
        ref_FFT = np.fft.fft(current_refTraces * window)
        dt = current_timeTrace[1] - current_timeTrace[0]
        ref_frq = np.fft.fftfreq(n, dt)
        positive_frq = ref_frq[:n // 2]
        positive_FFT = np.abs(ref_FFT[:n // 2])

        peak_indices, _ = find_peaks(positive_FFT, height=np.max(positive_FFT)/4)
        if len(peak_indices) == 0:
            continue

        dominant_freq = positive_frq[peak_indices[0]]
        width = 0.4

        lowF = dominant_freq - width / 2
        upF = dominant_freq + width / 2
        centre = (lowF + upF) / 2
        largeur = upF - lowF
        h2_lowF = 2 * centre - largeur / 2
        h2_upF = 2 * centre + largeur / 2

        plot_fft_with_band(
            signal=current_refTraces,
            timeTrace=current_timeTrace,
            lowF=lowF,
            upF=upF,
            h2_lowF=h2_lowF,
            h2_upF=h2_upF,
            pixel_coords="Reference",
            save_path=os.path.join(res_path, f"heatmaps/fft_reference_heatmap_{i+1}.png")
        )

        snr_map_opti = calculate_snr_map_opti(current_pulseTraces, current_timeTrace, lowF, upF)
        np.save(os.path.join(res_path, "npy_files", f"snr_map_{i}.npy"), snr_map_opti)

        if roi_coords:
            x, y, w, h = roi_coords
            roi_snrs = snr_map_opti[y:y+h, x:x+w]
            roi_mean_snr = np.mean(roi_snrs[np.isfinite(roi_snrs)])
            print(f"📌 Average SNR inside ROI_REF (x={x}, y={y}, w={w}, h={h}) for heatmap {i+1}: {roi_mean_snr:.2f} dB")

        average_snr = np.mean(snr_map_opti[np.isfinite(snr_map_opti)])
        print(f"📌 Average SNR for full frame (heatmap {i+1}): {average_snr:.2f} dB")

        cmap = cm.get_cmap('jet')
        fig, ax = plt.subplots()
        vmin = np.percentile(snr_map_opti, 5)   # 5th percentile
        vmax = np.percentile(snr_map_opti, 95)  # 95th percentile

        im = ax.imshow(snr_map_opti, cmap='jet', vmin=vmin, vmax=vmax)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins()
        plt.tight_layout()
        plt.savefig(os.path.join(res_path, "heatmaps", f'SNR_MAP_FRQ_{i}.png'), format='png', bbox_inches='tight', pad_inches=0.05, dpi=300)
        # Save only the hand heatmap without colorbar or padding
        fig2, ax2 = plt.subplots(figsize=(snr_map_opti.shape[1] / 100, snr_map_opti.shape[0] / 100), dpi=100)
        ax2.imshow(snr_map_opti, cmap='jet', vmin=vmin, vmax=vmax)
        ax2.axis('off')  # Hide axes
        fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
        save_clean_path = os.path.join(res_path, "heatmaps", f'SNR_MAP_NO_BAR_{i}.png')
        fig2.savefig(save_clean_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig2)
        plt.close()


def snr_map_opti(pixel_signal, lowF, upF, h2_lowF, h2_upF, centre, largeur, timeTrace):
    window = np.hanning(len(pixel_signal))
    windowed_signal = pixel_signal * window
    pulseTrace_FFT = np.fft.fft(windowed_signal)
    dt = timeTrace[1] - timeTrace[0]
    frq = np.fft.fftfreq(len(pixel_signal), dt)
    df = frq[1] - frq[0]                    # Define frequency resolution
    width_hz = 0.4                           # 0.3 Hz band around peaks
    power_spectrum = np.abs(pulseTrace_FFT) ** 2        # Find power spectrum
    peak1 = (lowF + upF) / 2            # Center frequency from input
    peak2 = 2 * peak1

    range1 = (frq >= peak1 - width_hz/2) & (frq <= peak1 + width_hz/2)      # Create signal masks around peaks ±width/2
    range2 = (frq >= peak2 - width_hz/2) & (frq <= peak2 + width_hz/2)
    signal_mask = range1 | range2
    positive_freqs = frq > 0            # Keep only positive frequencies
    signal_mask &= positive_freqs

    signal_power = np.sum(power_spectrum[signal_mask])      # Signal and noise power
    noise_power = np.sum(power_spectrum[positive_freqs & (~signal_mask)])

    snr_value = 10 * np.log10(signal_power / noise_power + 1e-10)  # add epsilon to avoid log(0)
    return snr_value

def calculate_snr_map_opti(current_pulseTraces, timeTrace, lowF, upF):
    m, n, t = current_pulseTraces.shape
    snr_map = np.zeros((m, n))
    centre = (lowF + upF) / 2
    largeur = upF - lowF
    h2_lowF = 2 * centre - largeur / 2
    h2_upF = 2 * centre + largeur / 2
    results = Parallel(n_jobs=os.cpu_count())(
        delayed(snr_map_opti)(current_pulseTraces[i, j], lowF, upF, h2_lowF, h2_upF, centre, largeur, timeTrace)
        for i in tqdm(range(m), desc="SNR Heatmap opti")
        for j in range(n)
    )
    snr_map = np.array(results).reshape((m, n))
    return snr_map

def test_snr_map(snr_origin, snr_opti, shape):
    for i in range(shape[0]):
        for j in range(shape[1]):
            assert snr_origin[i, j] == snr_opti[i, j], f"SNR not equal at {i}, {j} : {snr_origin[i,j]} != {snr_opti[i,j]}"
