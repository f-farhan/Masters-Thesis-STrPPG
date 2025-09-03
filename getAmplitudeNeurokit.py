"""
Description : Code pour générer la carte d'amplitude moyennée, avec la méthode Neurokits.
La fonction findpeaks de Neurokit est spécialisée pour les signaux PPG. Elle est beaucoup plus précise même dans un environnement bruyant, mais le temps de calcul est extrêmement long.

Définition manuelle de la vmax des cartes (Normalisation).
"""

import os
import numpy as np
import neurokit2 as nk
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from joblib import Parallel, delayed
from config import cf

# ============ Fonction Principal ============ 
def amplitude_heatmap_neurokit(res_path, pulseTraces, timeTrace, mean_bpm, num_heatmaps, refTraces=None):
    if cf.DEBUG == 1:
       pulseTraces, timeTrace, refTraces = (
            np.load(os.path.join(res_path,"npy_files/pulseTraces.npy")), 
            np.load(os.path.join(res_path,"npy_files/timeTrace.npy")), 
            np.load(os.path.join(res_path,"npy_files/rppg_ref.npy")))  
    
    total_time = np.max(timeTrace) - np.min(timeTrace)
    time_per_heatmap = total_time / num_heatmaps        # Divise le temps total par le nombre de heatmaps pour avoir le temps par heatmaps

    for i in range(num_heatmaps):# Calcul une carte par nombre de heatmaps
        start_time = i * time_per_heatmap
        end_time = start_time + time_per_heatmap

        mask = (timeTrace >= start_time) & (timeTrace < end_time) # Réalise un masque pour selectionné que le temps courant pour la heatmaps
        current_pulseTraces = pulseTraces[:, :, mask]
        current_refTraces = refTraces[mask]

        ppg_map = calculate_amplitude_neurokit_opti(current_pulseTraces,mean_bpm,current_refTraces)
        cmap = cm.get_cmap('jet') # Choix de la colormap jet

        fig, ax = plt.subplots()
        im = ax.imshow(ppg_map, cmap=cmap, origin='upper', aspect='equal', vmax=1) #<- À définir vmax
        # Ajustement des axes pour ajouter un peu d'espace
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_xticks([])   # Supprime les graduations de l'axe X
        ax.set_yticks([])   # Supprime les graduations de l'axe Y
        ax.margins()        # Ajoute des marges autour de l'image
        plt.tight_layout()
        plt.savefig(os.path.join(res_path, "heatmaps", f'{"AMPL_MAP_NEUROKIT"}_{i}.png'), bbox_inches='tight',  format='png',pad_inches=0.05,dpi=300)
        plt.close()

def calculate_amplitude_neurokit(current_pulseTraces,mean_bpm,current_refTraces):
    m, n, t = current_pulseTraces.shape
    ppg_map = np.zeros((m, n))
     
    for i in tqdm(range(m), desc='Neurokit Findpeaks Aplitude Heatmap'):
        for j in range(n):
            pixel_signal = np.nan_to_num(current_pulseTraces[i, j, :])          # Pour chaque pixel # Remplacer les infs et NaNs par des valeurs valides

            # interpolation à 1000Hz (dois oversamplé pour neurokit)
            fs_original = 30  # YB TODO à modifier 
            fs_new = 1000   

            t_original = np.arange(len(pixel_signal)) / fs_original  # Temps d'origine
            t_new = np.linspace(0, (len(pixel_signal) - 1) / fs_original, int(len(pixel_signal) * (fs_new / fs_original)))  # Temps interpolé

            
            interpolator = interp1d(t_original, pixel_signal, kind='linear')
            signal_interpolated = interpolator(t_new)

            clean_pulstrace = nk.ppg_clean(signal_interpolated, method='elgendi', sampling_rate=fs_new, heart_rate=mean_bpm)   
            # plt.plot(t_original, pixel_signal, label='Signal')
            # plt.plot(t_new, clean_pulstrace, label='Signal clean', linestyle='--')
            # plt.show()

            peaks = nk.ppg_findpeaks(clean_pulstrace, sampling_rate=fs_new, method="elgendi")      # Pour les pics du signal 
            peaks_indices = peaks["PPG_Peaks"]
            peaks_indices = np.array(peaks_indices)
            peaks_indices = peaks_indices.flatten()

            troughs = nk.ppg_findpeaks(-clean_pulstrace, sampling_rate=fs_new, method="elgendi")   # Pour les creux du signal 
            troughs_indices = troughs["PPG_Peaks"]
            troughs_indices = np.array(troughs_indices)
            troughs_indices = troughs_indices.flatten()

            if len(peaks_indices) != len(troughs_indices):              # Synchronisation entre pics et creux pour avoir le même nombre de pics entre les deux
                min_len = min(len(peaks_indices), len(troughs_indices))
                peaks_indices = peaks_indices[:min_len]
                troughs_indices = troughs_indices[:min_len]

            peaks_values = clean_pulstrace[peaks_indices]
            troughs_values = clean_pulstrace[troughs_indices]

            if len(peaks_values) != len(troughs_values):
                print("Les tableaux pics_values et creux_values ont des longueurs différentes.")

            amplitudes = []
            for k in range(len(peaks_values)):
                if k < len(troughs_values):
                    amplitude = peaks_values[k] - troughs_values[k]     # Différence entre les amplitudes des pics et des creux, pour avoir l'amplitude
                    amplitudes.append(amplitude)

            mean_amplitude = np.median(amplitudes)                      # Moyennage des valeurs par pixels
            ppg_map[i, j] = mean_amplitude

    return ppg_map



def neurokit_findpeaks_amplitude(current_pulseTrace,mean_bpm):
    """
    Calcule l'amplitude Neurokit
    Args : 
        current_pulseTrace (np.array) : dimension : largeur de la ROI x hauteur de la ROI x nombre d'image.
        mean_bpm int: BPM moyen
    return:
        mean_amplitude : moyen des valeur par pixel (largeur de la ROI x Hauteur de la ROI) 
    """
    pixel_signal = np.nan_to_num(current_pulseTrace)          # Pour chaque pixel # Remplacer les infs et NaNs par des valeurs valides
    # interpolation à 1000Hz (dois oversamplé pour neurokit)
    fs_original = 30  # YB TODO à modifier 
    fs_new = 1000   

    t_original = np.arange(len(pixel_signal)) / fs_original  # Temps d'origine
    t_new = np.linspace(0, (len(pixel_signal) - 1) / fs_original, int(len(pixel_signal) * (fs_new / fs_original)))  # Temps interpolé

    
    interpolator = interp1d(t_original, pixel_signal, kind='linear')
    signal_interpolated = interpolator(t_new)

    clean_pulstrace = nk.ppg_clean(signal_interpolated, method='elgendi', sampling_rate=fs_new, heart_rate=mean_bpm)   

    peaks = nk.ppg_findpeaks(clean_pulstrace, sampling_rate=fs_new, method="elgendi")      # Pour les pics du signal 
    peaks_indices = peaks["PPG_Peaks"]
    peaks_indices = np.array(peaks_indices)
    peaks_indices = peaks_indices.flatten()

    troughs = nk.ppg_findpeaks(-clean_pulstrace, sampling_rate=fs_new, method="elgendi")   # Pour les creux du signal 
    troughs_indices = troughs["PPG_Peaks"]
    troughs_indices = np.array(troughs_indices)
    troughs_indices = troughs_indices.flatten()

    if len(peaks_indices) != len(troughs_indices):              # Synchronisation entre pics et creux pour avoir le même nombre de pics entre les deux
        min_len = min(len(peaks_indices), len(troughs_indices))
        peaks_indices = peaks_indices[:min_len]
        troughs_indices = troughs_indices[:min_len]

    peaks_values = clean_pulstrace[peaks_indices]
    troughs_values = clean_pulstrace[troughs_indices]

    if len(peaks_values) != len(troughs_values):
        print("Les tableaux pics_values et creux_values ont des longueurs différentes.")

    amplitudes = []
    for k in range(len(peaks_values)):
        if k < len(troughs_values):
            amplitude = peaks_values[k] - troughs_values[k]     # Différence entre les amplitudes des pics et des creux, pour avoir l'amplitude
            amplitudes.append(amplitude)

    mean_amplitude = np.median(amplitudes)                      # Moyennage des valeurs par pixels
    return mean_amplitude


def calculate_amplitude_neurokit_opti(current_pulseTraces,mean_bpm,current_refTraces):
    """
    Encapsulation de la fonction neurokit_findpeaks_amplitude avec JobLib
    Args:       
        current_pulseTrace (np.array) : dimension : largeur de la ROI x hauteur de la ROI x nombre d'image.
        mean_bpm int: BPM moyen
        current_refTraces (nd.array) : trace de reference
    return:
        mean_amplitude : moyen des valeur par pixel (largeur de la ROI x Hauteur de la ROI) 
    """
    m, n, t = current_pulseTraces.shape
    ppg_map = np.zeros((m, n))

    results = Parallel(n_jobs=os.cpu_count())(                               # Declare autant the Job que de processeur
        delayed(neurokit_findpeaks_amplitude)(current_pulseTraces[i,j], mean_bpm)                  
        for i in tqdm(range(m), desc="Neurokit Finpeaks Amplitude Heatmap Opti ")
        for j in range(n)
    )
    ppg_map = np.array(results).reshape(m,n)
    return ppg_map

def test_neurokit_equality(ppg_map, ppg_map2):
    """
    Compare les les ppg maps avec la fonction original et la version opti
    """
    for i in tqdm(range(m), desc="testing Neurokit equality"):
        for j in range(n):
            assert(ppg_map[i,j]== ppg_map2[i,j]), f"PPG map diferents at {i}, {j} : expected {ppg_map[i,j]} got {ppg_map2[i,j]}"
