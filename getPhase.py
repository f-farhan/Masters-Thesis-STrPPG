"""
Description : Code pour générer la carte de phase moyenné, avec la méthode Hilbert. On utilise la transformée de Hilbert pour obtenir le signal analytique (un signal complexe)
On calcule l'angle complexe entre le signal analytique du signal d'entrée et celui des traces de référence pour obtenir le déphasage de phase entre le signal par pixel et le signal de référence



Définition manuelle de la vmax des cartes (Normalisation).
Nous avons mis -90° et 90° alors que -180° et 180° aurait été plus judicieux, car visuellement avec nos résultats, le rendu avec la colormap était meilleur.
"""

import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.signal import hilbert
from mpl_toolkits.axes_grid1 import make_axes_locatable

from joblib import Parallel, delayed
from config import cf
# ============ Fonction Principal ============
def phase_heatmap_hilbert(res_path, pulseTraces, timeTrace, num_heatmaps, refTraces=None):
    if cf.DEBUG == 1:
       pulseTraces, timeTrace, refTraces = (
            np.load(os.path.join(res_path,"npy_files/pulseTraces.npy")), 
            np.load(os.path.join(res_path,"npy_files/timeTrace.npy")), 
            np.load(os.path.join(res_path,"npy_files/rppg_ref.npy")))
       

    total_time = np.max(timeTrace) - np.min(timeTrace)
    time_per_heatmap = total_time / num_heatmaps            # Divise le temps total par le nombre de heatmaps pour avoir le temps par heatmaps

    for i in range(num_heatmaps):    # Calcul une carte par nombre de heatmaps
        start_time = i * time_per_heatmap
        end_time = start_time + time_per_heatmap

    
        mask = (timeTrace >= start_time) & (timeTrace < end_time) # Réalise un masque pour selectionné que le temps courant pour la heatmaps
        current_pulseTraces = pulseTraces[:, :, mask]
        current_timeTrace = timeTrace[mask]
        current_refTraces = refTraces[mask]
        
        #created_refTraces = creation_Reference(current_refTraces,timeTrace) # Crée un signal de référence synthétique

        phase_map_opti = calculate_phase_map_opti(current_pulseTraces, current_refTraces)  # Fonction de calcul de la carte de phase.
        #phase_map = calculate_phase_map(current_pulseTraces, current_refTraces)  # Fonction de calcul de la carte de phase.
        #test_hilbert(phase_map, phase_map_opti, current_pulseTraces.shape)
        create_phase_heatmap(phase_map_opti, os.path.join(res_path, "heatmaps", f"PHASE_MAP_Hilbert{i}.png")) #code changed


# ============ Fonction Calcul ============
def calculate_phase_map(ppg_signal, current_refTraces):
    m, n, _ = ppg_signal.shape
    phase_map = np.zeros((m, n))

    for i in tqdm(range(m), desc='Phase Hilbert Heatmap'):
        for j in range(n):
            pixel_signal = ppg_signal[i, j]
            if np.all(pixel_signal == 0):
                phase_map[i, j] = np.nan
            else:
                phase = calculate_phase(pixel_signal, current_refTraces)
                phase_map[i, j] = np.mean(phase) # On fait la moyenne sur la totalité du signal compris dans la période encours.

    return phase_map


# ============ Fonction de calcul du déphasage de phase ============
def calculate_phase(signal, ref_traces):
    
    analytic_signal = hilbert(signal)
    analytic_ref_traces = hilbert(ref_traces)

    # Calculer le déphasage de phase (en degrés)
    phase = np.angle(analytic_signal / analytic_ref_traces, deg=True)

    return phase


# ============ Fonction Création Heatmap ============
def create_phase_heatmap(phase_map, buf):
    cmap = cm.get_cmap('seismic') 
    fig, ax = plt.subplots()
    im = ax.imshow(phase_map, cmap=cmap, origin='upper', aspect='equal', vmin=-180, vmax=180)  #<- À définir La valeurs de phases sont entre -90° et 90°
    # Ajustement des axes pour ajouter un peu d'espace
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Phase deg')
    ax.set_xticks([])   # Supprime les graduations de l'axe X
    ax.set_yticks([])   # Supprime les graduations de l'axe Y
    ax.margins()        # Ajoute des marges autour de l'image
    plt.tight_layout()
    plt.savefig(buf, format='png',bbox_inches='tight',pad_inches=0.05,dpi=300)
    plt.close()

def creation_Reference(window_signal, temps): # Création d'un signal de référence synthétique 
    hr_bpm = 64.58  # <- À définir
    wHR = 2 * np.pi * (hr_bpm / 60)
    t = np.linspace(temps[0], temps[-1], len(window_signal))
    reference_signal = np.cos(wHR * t) + 1j * np.sin(wHR * t)
    return reference_signal

def calculate_phase_opti(pixel_signal, current_refTraces):
    pixel_signal = np.where(pixel_signal==0, np.nan, pixel_signal)
    analytic_signal = hilbert(pixel_signal)
    analytic_ref_traces = hilbert(current_refTraces)
    phase = np.angle(analytic_signal / analytic_ref_traces, deg=True)
    return np.mean(phase) # On fait la moyenne sur la totalité du signal compris dans la période encours.

def calculate_phase_map_opti(ppg_signal, current_refTraces):
    m, n, _ = ppg_signal.shape
    phase_map = np.zeros((m, n))

    results = Parallel(n_jobs=os.cpu_count())(
        delayed(calculate_phase_opti)(ppg_signal[i,j], current_refTraces)
        for i in tqdm(range(m), desc="Phase Hilbert Heatmap Opti")
        for j in range(n)
    ) 
    phase_map = np.array(results).reshape((m,n))
    return phase_map

def test_hilbert(phase_map, phase_map_opti, shape):
    for i in range(shape[0]):
        for j in range(shape[1]):
            assert(phase_map[i,j]==phase_map_opti[i,j]), f"Hilbert map not equal at index {i}, {j} : {phase_map[i,j]} != {phase_map_opti[i,j]}"
