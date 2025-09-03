"""
Description : Code pour générer la carte d'amplitude moyennée, avec la méthode FINDPEAKS.
FINDPEAKS détecte les pics et les creux dans un signal. On peut ainsi faire la moyenne des variations dans le temps pour obtenir une valeur par pixel.

Définition manuelle de la vmax des cartes (Normalisation),
Définition manuelle des paramètres de FINDPEAKS,
ce qui peut ne pas fonctionner dans certains cas, en fonction de la forme du signal. À adapter selon les situations.
"""

import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from scipy.signal import  find_peaks
from mpl_toolkits.axes_grid1 import make_axes_locatable
from joblib import Parallel, delayed
from config import cf

# ============ Fonction Principal ============ 
def amplitude_heatmap_FINDPEAKS(res_path, pulseTraces, timeTrace,refTraces, num_heatmaps):

    if cf.DEBUG == 1:
        pulseTraces, timeTrace, refTraces = (
        np.load(os.path.join(res_path,"npy_files/pulseTraces.npy")), 
        np.load(os.path.join(res_path,"npy_files/timeTrace.npy")), 
        np.load(os.path.join(res_path,"npy_files/rppg_ref.npy")))   


    total_time = np.max(timeTrace) - np.min(timeTrace) 
    time_per_heatmap = total_time / num_heatmaps                    # Divise le temps total par le nombre de heatmaps pour avoir le temps par heatmaps

    for i in range(num_heatmaps):            # Calcul une carte par nombre de heatmaps
        start_time = i * time_per_heatmap
        end_time = start_time + time_per_heatmap

        mask = (timeTrace >= start_time) & (timeTrace < end_time)   # Réalise un masque pour selectionné que le temps courant pour la heatmaps
        current_pulseTraces = pulseTraces[:, :, mask]
        current_refTraces = refTraces[mask]

        amp_map = calculate_amplitude_findpeaks_opti(current_pulseTraces,current_refTraces)
        #amp_map_origin = calculate_amplitude_findpeaks(current_pulseTraces,current_refTraces)
        #test_findpeaks(amp_map_origin, amp_map)
        cmap = cm.get_cmap('jet')                                   # Choix de la colormap jet

        fig, ax = plt.subplots()
        im = ax.imshow(amp_map, cmap=cmap, origin='upper', aspect='equal', vmax=0.8) #<- À définir vmax
        # Ajustement des axes pour ajouter un peu d'espace
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_xticks([])   # Supprime les graduations de l'axe X
        ax.set_yticks([])   # Supprime les graduations de l'axe Y
        ax.margins()        # Ajoute des marges autour de l'image
        plt.tight_layout()
        plt.savefig(os.path.join(res_path, "heatmaps", f'{"AMPL_MAP_FINDPEAKS"}_{i}.png'), bbox_inches='tight',  format='png',pad_inches=0.05,dpi=300)
        plt.close()

        
# ============ Fonction de calcul de l'amplitude ============
def calculate_amplitude_findpeaks(current_pulseTraces, current_refTraces):
    m, n, t = current_pulseTraces.shape
    amp_map = np.zeros((m, n))

    for i in tqdm(range(m), desc='FINDPEAKS'): 
        for j in range(n):
            pixel_signal = np.nan_to_num(current_pulseTraces[i, j, :])                      # Remplacer les infs et NaNs par des valeurs valides

            peaks, properties = find_peaks(pixel_signal, height=0, distance=8)              # Trouver les pics  <- À définir  height, distance, etc 
            troughs, trough_properties = find_peaks(-pixel_signal, height=0, distance=8)    # Trouver les creux <- À définir  height, distance, etc 

            amplitudes = []
            for peak, peak_height in zip(peaks, properties['peak_heights']):
                if len(troughs) == 0:                                                       # S'il n'y a pas de creux
                    amplitude = peak_height                                                 # L'amplitude est simplement la hauteur du pic
                else:
                    closest_trough = min(troughs, key=lambda trough: abs(trough - peak))    # On fait correspondre le creux au pic le plus proche
                    trough_height = -trough_properties['peak_heights'][list(troughs).index(closest_trough)]
                    amplitude = peak_height - trough_height                                 # La difference des deux correspond à l'amplitude creux-pic (Prend en compte les valeurs positives et négatives)
                amplitudes.append(amplitude)

            mean_amplitude = np.median(amplitudes)                                          # Moyennage des valeurs
            if mean_amplitude <= 0:  
                mean_amplitude = np.nan                                                     # Suppression du fond, enlever le valeur null

            amp_map[i, j] = mean_amplitude

    return amp_map

def find_peaks_opti(pixel_signal, current_refTraces):
    pixel_signal = np.nan_to_num(pixel_signal)
    peaks, properties = find_peaks(pixel_signal, height=0, distance=8)              # Trouver les pics  <- À définir  height, distance, etc 
    troughs, trough_properties = find_peaks(-pixel_signal, height=0, distance=8)    # Trouver les creux <- À définir  height, distance, etc 

    amplitudes = []
    for peak, peak_height in zip(peaks, properties['peak_heights']):
        if len(troughs) == 0:                                                       # S'il n'y a pas de creux
            amplitude = peak_height                                                 # L'amplitude est simplement la hauteur du pic
        else:
            closest_trough = min(troughs, key=lambda trough: abs(trough - peak))    # On fait correspondre le creux au pic le plus proche
            trough_height = -trough_properties['peak_heights'][list(troughs).index(closest_trough)]
            amplitude = peak_height - trough_height                                 # La difference des deux correspond à l'amplitude creux-pic (Prend en compte les valeurs positives et négatives)
        amplitudes.append(amplitude)

    mean_amplitude = np.median(amplitudes)                                          # Moyennage des valeurs
    if mean_amplitude <= 0:  
        mean_amplitude = np.nan                                                     # Suppression du fond, enlever le valeur null

    return mean_amplitude


def calculate_amplitude_findpeaks_opti(current_pulseTraces, current_refTraces):
    m, n, t = current_pulseTraces.shape
    amp_map = np.zeros((m, n))

    results = Parallel(n_jobs=os.cpu_count())(
        delayed(find_peaks_opti)(current_pulseTraces[i,j], current_refTraces)
        for i in tqdm(range(m), desc = "FindPeaks opti")
        for j in range(n)    
    )
    amp_map = np.array(results).reshape((m,n))
    return amp_map
    
def test_findpeaks(origin, opti):
    assert(np.array_equal(origin, opti))
