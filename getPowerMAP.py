"""
Description : Code pour générer la carte de puissance moyenne par pixel avec la FFT.



Définition manuelle de la vmax des cartes (Normalisation).
"""

import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from config import cf

# ============ Fonction Principal ============ 
def power_map(res_path, pulseTraces):

    if cf.DEBUG == 1:
        pulseTraces = np.load(os.path.join(res_path,"npy_files/pulseTraces.npy"))
            
        power_map = calculate_power_map(pulseTraces)

        # Enregistrement de la carte de puissance
        fig, ax = plt.subplots()
        cmap = cm.get_cmap('hot')
        im = ax.imshow(power_map, cmap=cmap, origin='upper', aspect='equal', vmax=2000)
        
        # Ajustement des axes pour ajouter un peu d'espace
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        ax.set_xticks([])   # Supprime les graduations de l'axe X
        ax.set_yticks([])   # Supprime les graduations de l'axe Y
        ax.margins()        # Ajoute des marges autour de l'image
        
        plt.tight_layout()
        plt.savefig(os.path.join(res_path, "heatmaps", f"POWER_MAP_.png"), format='png', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

# ============ Fonction de calcul de la power map ============
def calculate_power_map(ppg_signal):
    # Initialisation de la carte de puissance
    m, n, _ = ppg_signal.shape
    power_map = np.zeros((m, n))
    
    # Calcul de la puissance maximale pour chaque pixel
    for i in tqdm(range(m), desc='POWER'):
        for j in range(n):
            pixel_signal = ppg_signal[i, j, :]          # Extrait le signal du pixel à la position (i, j)
            pulseTraceFFT = np.fft.fft(pixel_signal)    # Calcule la transformée de Fourier rapide du signal du pixel
            power_spectrum = np.abs(pulseTraceFFT)**2   # Calcule le spectre de puissance (amplitude au carré des composantes de Fourier)
            power_map[i, j] = np.max(power_spectrum)    # Stocke la puissance maximale du spectre dans la carte de puissance
    
    return power_map
