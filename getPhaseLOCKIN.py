"""
Description : Code pour générer la carte de phase moyenne, avec la méthode LOCKIN.
Le calcul de la phase se fait suite à la multiplication entre le signal de référence et le signal par pixel. 
Ensuite, on calcule l'arc tangent de la partie imaginaire par rapport à la partie réelle du signal multiplié.

Il est possible d'utiliser le signal de référence pour la multiplication au lieu de générer un signal synthétique.

Le résultat de cette méthode est étonnamment moins bon qu'avec la méthode Hilbert.
 
Définition manuelle de la vmax des cartes (Normalisation).
Nous avons choisi -90° et 90° alors que -180° et 180° auraient été plus judicieux, car visuellement, avec nos résultats, le rendu avec la colormap était meilleur.
"""
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, hilbert
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from joblib import Parallel, delayed
from config import cf
# ============ Fonction Principal ============ 
def phase_heatmap_LOCKIN(res_path, pulseTraces, timeTrace,mean_bpm,refTraces, num_heatmaps):

    if cf.DEBUG == 1:
         pulseTraces, timeTrace, refTraces = (
            np.load(os.path.join(res_path,"npy_files/pulseTraces.npy")), 
            np.load(os.path.join(res_path,"npy_files/timeTrace.npy")), 
            np.load(os.path.join(res_path,"npy_files/rppg_ref.npy")))
         
    total_time = np.max(timeTrace) - np.min(timeTrace)
    time_per_heatmap = total_time / num_heatmaps # Divise le temps total par le nombre de heatmaps pour avoir le temps par heatmaps

    for i in range(num_heatmaps): # Calcul une carte par nombre de heatmaps
        start_time = i * time_per_heatmap
        end_time = start_time + time_per_heatmap

        mask = (timeTrace >= start_time) & (timeTrace < end_time) # Réalise un masque pour selectionné que le temps courant pour la heatmaps
        current_pulseTraces = pulseTraces[:, :, mask]
        current_refTraces = refTraces[mask]
        

        pha_map = calculate_map_phase_opti(current_pulseTraces,timeTrace,mean_bpm,current_refTraces) # Fonction de calcul de la carte de phase.
        #pha_mapi_origin = calculate_map_phase(current_pulseTraces,timeTrace,mean_bpm,current_refTraces) # Fonction de calcul de la carte de phase.
        #test_phase_lockin(pha_mapi_origin, pha_map, current_pulseTraces.shape)
        cmap = cm.get_cmap('seismic')                                                           # Colormap
        
        fig, ax = plt.subplots()
        im = ax.imshow(pha_map, cmap=cmap, origin='upper', aspect='equal', vmin=-180, vmax=180) #<- À définir La valeurs de phases sont entre -90° et 90°
        # Ajustement des axes pour ajouter un peu d'espace
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_xticks([])   # Supprime les graduations de l'axe X
        ax.set_yticks([])   # Supprime les graduations de l'axe Y
        ax.margins()        # Ajoute des marges autour de l'image
        plt.tight_layout()
        plt.savefig(os.path.join(res_path, "heatmaps", f'{"PHASE_MAP_LOCKIN"}_{i}.png'), format='png',bbox_inches='tight',pad_inches=0.05,dpi=300)
        plt.close()

        
# ============ Fonction pour le multiprocess ============ 
def calculate_map_phase(current_pulseTraces,timeTrace,mean_bpm,current_refTraces):
    m, n, t = current_pulseTraces.shape
    pha_map = np.zeros((m, n))
    coefCorr = np.zeros((m, n))

    for i in tqdm(range(m), desc='PHASE LOCKIN'):
        for j in range(n):
            pha_map[i, j] = calculate_pixel (i, j, current_pulseTraces,timeTrace,mean_bpm,current_refTraces,coefCorr)[2]

    return pha_map


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # Fréquence de Nyquist
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# ============ Fonction de calcul de l'amplitude ============
def calculate_pixel(i, j, current_pulseTraces,timeTrace,mean_bpm,current_refTraces,coefCorr):
    pixel_signal = np.nan_to_num(current_pulseTraces[i, j, :]) #Pour chaque pixel # Remplacer les infs et NaNs par des valeurs valides
   
    # ============ LOCKIN ============
    analytic_reference_signal = hilbert(pixel_signal)
    reference_signal_quadrature = np.imag(analytic_reference_signal)  # Composante en quadrature (90°)

    signal_in_phase = pixel_signal * current_refTraces
    signal_in_quadrature = pixel_signal * reference_signal_quadrature

    cutoff = 3
    time_diffs = np.diff(timeTrace)
    mean_time_diff = np.mean(time_diffs)
    fs = 1 / mean_time_diff

    filtered_signal_in_phase = lowpass_filter(signal_in_phase, cutoff, fs)
    filtered_signal_in_quadrature = lowpass_filter(signal_in_quadrature, cutoff, fs)

    phase = np.arctan2(filtered_signal_in_quadrature, filtered_signal_in_phase) * 180 / np.pi  # Phase en degrés

    mean_phase = np.mean(phase)
    
    return i, j, mean_phase

def calculate_pixel_opti(pixel_signal,timeTrace,mean_bpm,current_refTraces,coefCorr):
    analytic_reference_signal = hilbert(pixel_signal)
    reference_signal_quadrature = np.imag(analytic_reference_signal)  # Composante en quadrature (90°)

    signal_in_phase = pixel_signal * current_refTraces
    signal_in_quadrature = pixel_signal * reference_signal_quadrature

    cutoff = 3
    time_diffs = np.diff(timeTrace)
    mean_time_diff = np.mean(time_diffs)
    fs = 1 / mean_time_diff

    filtered_signal_in_phase = lowpass_filter(signal_in_phase, cutoff, fs)
    filtered_signal_in_quadrature = lowpass_filter(signal_in_quadrature, cutoff, fs)

    phase = np.arctan2(filtered_signal_in_quadrature, filtered_signal_in_phase) * 180 / np.pi  # Phase en degrés

    mean_phase = np.mean(phase)
    
    return mean_phase

def calculate_map_phase_opti(current_pulseTraces,timeTrace,mean_bpm,current_refTraces):
    m, n, t = current_pulseTraces.shape
    pha_map = np.zeros((m, n))
    coefCorr = np.zeros((m, n))

    results = Parallel(n_jobs=os.cpu_count())(
        delayed(calculate_pixel_opti)(current_pulseTraces[i,j], timeTrace, mean_bpm, current_refTraces, coefCorr)
        for i in tqdm(range(m), desc="Phase Lockin Opti")
        for j in range(n)
    )

    pha_map = np.array(results).reshape((m,n))
    return pha_map


def test_phase_lockin(pha_map_origin, pha_map_opti, shape):
    for i in range(shape[0]):
        for j in range(shape[1]):
            assert(pha_map_origin[i,j]==pha_map_opti[i,j]),f"Pha_map not equal at index {i}, {j} : {pha_map_origin[i,j]} != {pha_map_opti[i,j]}" 
