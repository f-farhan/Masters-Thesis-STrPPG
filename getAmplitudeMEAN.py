"""
Description : Code pour générer la carte d'amplitude moyennée, avec la méthode MEAN.
C'est la méthode la plus simple et la plus rapide pour récupérer l'amplitude moyenne d'un pixel.
Elle prend en compte la moyenne des valeurs absolues, car les signaux sont normalisés. 

Définition manuelle de la vmax des cartes (Normalisation).
"""

import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from config import cf

# ============ Fonction Principale  ============
def amplitude_heatmap_mean(res_path, pulseTraces, timeTrace, num_heatmaps):

    if cf.DEBUG == 1:
        pulseTraces, timeTrace = (
        np.load(os.path.join(res_path,"npy_files/pulseTraces.npy")), 
        np.load(os.path.join(res_path,"npy_files/timeTrace.npy"))) 

    total_time = np.max(timeTrace) - np.min(timeTrace)
    time_per_heatmap = total_time / num_heatmaps        # Divise le temps total par le nombre de heatmaps pour avoir le temps par heatmaps
    
    for i in range(num_heatmaps):# Calcul une carte par nombre de heatmaps
        start_time = i * time_per_heatmap
        end_time = start_time + time_per_heatmap 

        
        mask = (timeTrace >= start_time) & (timeTrace < end_time)   # Réalise un masque pour selectionné que le temps courant pour la heatmaps
        m, n, _ = pulseTraces.shape
        amp_map = np.zeros((m, n))                                  # Initialisation de la carte d'amplitude

        for i in tqdm(range(m), desc='MEAN'):
            for j in range(n):
                windowed_signal = pulseTraces[i, j, mask]           # Application du masque pour isoler le signal dans la fenêtre de temps
                amp_map[i, j] = np.mean(np.abs(windowed_signal))    # Moyenne des valeurs absolues
                
                if  amp_map[i, j] == 0:                             # Suppression du fond, enlever le valeur null
                    amp_map[i, j] = np.nan

        # Affichage de la heatmap
        cmap = cm.get_cmap('jet')
        fig, ax = plt.subplots()
        im = ax.imshow(amp_map, cmap=cmap, origin='upper', aspect='equal', vmax=0.2) #<- À définir vmax
        # Ajustement des axes pour ajouter un peu d'espace
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_xticks([])   # Supprime les graduations de l'axe X
        ax.set_yticks([])   # Supprime les graduations de l'axe Y
        ax.margins()        # Ajoute des marges autour de l'image
        plt.tight_layout()
        plt.savefig(os.path.join(res_path, "heatmaps", f'MEAN_AMPL_MAP.png'), bbox_inches='tight', format='png', pad_inches=0.05, dpi=300)
        plt.close()
