"""
Description : Code pour générer la carte d'amplitude moyennée, avec la méthode LOCKIN.
On multiplie notre signal par pixel par un signal de référence et, en calculant la racine carrée des parties réelle et imaginaire, on obtient l'amplitude désirée. 


Définition manuelle de la vmax des cartes (Normalisation).
Choix manuel de multiplication par le signal de référence ou par un signal synthétique.
"""
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import butter, filtfilt, hilbert

from joblib import Parallel, delayed
from config import cf

# ============ Fonction Principal ============ 
def amplitude_heatmap_LOCKIN(res_path, pulseTraces, timeTrace,mean_bpm,refTraces, num_heatmaps):

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

    
        amp_map = calculate_amplitude_LOCKIN_opti(current_pulseTraces,timeTrace,mean_bpm,current_refTraces) # Fonction de calcul de la carte d'amplitude
        #amp_map = calculate_amplitude_LOCKIN(current_pulseTraces,timeTrace,mean_bpm,current_refTraces) # Fonction de calcul de la carte d'amplitude
        #test_phase_lockin(amp_map, amp_map_opti, current_pulseTraces.shape)
        cmap = cm.get_cmap('jet') ## Choix de la colormap jet
        
        ##### FILTRAGE PAR HISTOGRAMME
        
        amp_bis = np.copy(amp_map)                                      # Copies des résultats bruts
        amplitudes_aplaties = amp_map.flatten()                         # Aplatir le tableau en 1D
        if np.isnan(amplitudes_aplaties).any():                         # Si des NaN sont présents, remplacez-les par des zéros ou traitez-les selon votre logique
            amplitudes_aplaties[np.isnan(amplitudes_aplaties)] = 0      # Remplacez les NaN par 0

        hist, bin_edges = np.histogram(amplitudes_aplaties, bins='auto')# Réalisation de l'histogramme

        indices_bins_à_conserver = np.where(hist >= 50)[0]              # Trouver les indices des bins avec une fréquence ≥ 50
        masque = np.zeros_like(amplitudes_aplaties, dtype=bool)
        for y in indices_bins_à_conserver:                              # Mettre à jour le masque pour True pour les pixels à conserver
            masque |= ((amplitudes_aplaties >= bin_edges[y]) & (amplitudes_aplaties < bin_edges[y+1]))

        masque_à_remplacer = ~masque                                    # Inverser le masque pour identifier les pixels à remplacer
        amplitudes_aplaties_modifiées = amplitudes_aplaties.copy()
        amplitudes_aplaties_modifiées[masque_à_remplacer] = 0           # Remplacer les valeurs indésirables par 0
        
        #### Pour supprimer les pixels inaproprié #####
        #amplitudes_filtrées = amplitudes_aplaties[np.any([((amplitudes_aplaties >= bin_edges[i]) & (amplitudes_aplaties < bin_edges[i+1])) for i in range(len(indices_bins_à_conserver))], axis=0)]
        
        # Histogramme amplitudes filtrées
        # plt.figure()
        # plt.hist(amplitudes_aplaties_modifiées, bins="auto",edgecolor='black')
        # plt.title('Histogramme des Amplitudes avec Binning et Filtrage')
        # plt.xlabel('Amplitude')
        # plt.ylabel('Fréquence')
        
        # Histogramme amplitudes brutes
        # plt.figure() 
        # plt.hist(amplitudes_aplaties, bins='auto', edgecolor='black')
        # plt.title('Histogramme des Amplitudes Brutes')
        # plt.xlabel('Amplitude')
        # plt.ylabel('Fréquence')
        # plt.show()

        #NORMALISATION
        amp_bis_height, amp_bis_width = amp_bis.shape
        normalized_amp_map = amplitudes_aplaties_modifiées.reshape(amp_bis_height, amp_bis_width)
        normalized_amp_map = normalized_amp_map / normalized_amp_map.max()
        
        
        #normalized_amp_map = amp_map / 0.005 # Normalisation Basique
              
        # CREATION CARTE AMP 
        fig, ax = plt.subplots()
        im = ax.imshow(normalized_amp_map, cmap=cmap, origin='upper', aspect='equal', vmax=1) #<- À définir vmax
        # Ajustement des axes pour ajouter un peu d'espace
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_xticks([])   # Supprime les graduations de l'axe X
        ax.set_yticks([])   # Supprime les graduations de l'axe Y
        ax.margins()        # Ajoute des marges autour de l'image
        plt.tight_layout()
        plt.savefig(os.path.join(res_path, "heatmaps", f'{"AMPL_MAP_LOCKIN_NORM"}_{i}.png'), bbox_inches='tight',  format='png',pad_inches=0.05,dpi=300)
        plt.close() 
#PLT 
#PLT 
        fig, ax = plt.subplots()
        im = ax.imshow(amp_map/np.max(amp_map), cmap=cmap, origin='upper', aspect='equal', vmax=0.01) #<- À définir vmax
        # Ajustement des axes pour ajouter un peu d'espace
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_xticks([])   # Supprime les graduations de l'axe X
        ax.set_yticks([])   # Supprime les graduations de l'axe Y
        ax.margins()        # Ajoute des marges autour de l'image
        plt.tight_layout()
        plt.savefig(os.path.join(res_path, "heatmaps", f'{"AMPL_MAP_LOCKIN"}_{i}.png'), bbox_inches='tight',  format='png',pad_inches=0.05,dpi=300)
        plt.close() 

               
# ============ Fonction de calcul de l'amplitude par pixel ============ 
def calculate_amplitude_LOCKIN(current_pulseTraces,timeTrace,mean_bpm,current_refTraces):
    m, n, t = current_pulseTraces.shape
    amp_map = np.zeros((m, n))
    for i in tqdm(range(m), desc='AMP LOCKIN'):  # Ajout de tqdm pour la barre de progression  
        for j in range(n):
            i, j, median_peak_amplitude = calculate_pixel(i, j, current_pulseTraces,timeTrace,mean_bpm,current_refTraces)
        
            amp_map[i, j] = median_peak_amplitude
            if amp_map[i, j] == 0:
                amp_map[i, j] = np.nan
    return amp_map

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # Fréquence de Nyquist
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# ============ LOCKIN ============
def calculate_pixel(i, j, current_pulseTraces,timeTrace,mean_bpm,current_refTraces):
    pixel_signal = np.nan_to_num(current_pulseTraces[i, j, :])                              # Pour chaque pixel Remplacer les infs et NaNs par des valeurs valides                                             # Moyennage de l'amplitude 

    analytic_reference_signal = hilbert(current_refTraces)
    reference_signal_quadrature = np.imag(analytic_reference_signal)  # Composante en quadrature (90°)
    signal_in_phase = pixel_signal * current_refTraces
    signal_in_quadrature = pixel_signal * reference_signal_quadrature

    time_diffs = np.diff(timeTrace)
    mean_time_diff = np.mean(time_diffs)
    fs = 1 / mean_time_diff

    cutoff = 3 
    filtered_signal_in_phase = lowpass_filter(signal_in_phase, cutoff, fs)
    filtered_signal_in_quadrature = lowpass_filter(signal_in_quadrature, cutoff, fs)
    amplitude = np.sqrt(filtered_signal_in_phase**2 + filtered_signal_in_quadrature**2)
    mean_amplitude = np.mean(amplitude)

    return i, j, mean_amplitude

def calculate_pixel_opti(pixel_signal,timeTrace,mean_bpm,current_refTraces):
    analytic_reference_signal = hilbert(current_refTraces)
    reference_signal_quadrature = np.imag(analytic_reference_signal)  # Composante en quadrature (90°)
    signal_in_phase = pixel_signal * current_refTraces
    signal_in_quadrature = pixel_signal * reference_signal_quadrature

    time_diffs = np.diff(timeTrace)
    mean_time_diff = np.mean(time_diffs)
    fs = 1 / mean_time_diff

    cutoff = 3 
    filtered_signal_in_phase = lowpass_filter(signal_in_phase, cutoff, fs)
    filtered_signal_in_quadrature = lowpass_filter(signal_in_quadrature, cutoff, fs)
    amplitude = np.sqrt(filtered_signal_in_phase**2 + filtered_signal_in_quadrature**2)
    mean_amplitude = np.mean(amplitude)

    return mean_amplitude

def calculate_amplitude_LOCKIN_opti(current_pulseTraces,timeTrace,mean_bpm,current_refTraces):
    m, n, t = current_pulseTraces.shape
    amp_map = np.zeros((m, n))

    results = Parallel(n_jobs=os.cpu_count())(
        delayed(calculate_pixel_opti)(current_pulseTraces[i,j], timeTrace, mean_bpm, current_refTraces)
        for i in tqdm(range(m), desc="Amplitude Lockin opti")
        for j in range(n)
    )
    mean_amplitude = np.array(results).reshape((m,n))
    mean_amplitude = np.where(mean_amplitude==0, np.nan, mean_amplitude)
    
    return mean_amplitude

def test_phase_lockin(mean_amp_original, mean_amp_opti, shape):
    for i in range(shape[0]):
        for j in range(shape[1]):
            assert(mean_amp_original[i,j]== mean_amp_opti[i,j]), f"Mean Amp not equal at index {i}, {j} : {mean_amp_original[i,j]} != {mean_amp_opti[i,j]}"
