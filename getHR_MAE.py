"""
Description : Code pour générer la carte du MAE (Mean Absolute Error) du BPM avec la méthode FFT. On calcule la différence entre le BPM trouvé dans chaque pixel et le BPM moyen, calculé via le signal de référence.

Attention à la taille de la fenêtre du HR, définie manuellement, qui peut changer drastiquement les résultats.
Pour être beaucoup plus précis au niveau de la carte, il est recommandé de faire la différence avec le BPM moyen de l'oxymètre de pouls et non avec celui de la zone de référence, qui correspond à la zone traitée.
"""
import os
import math
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

#from ARCHIVE_getHR import FFT
from getHR import BPM_FFT_simple
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from joblib import Parallel, delayed
import time 
from config import cf

def mae_heatmap(res_path, pulseTraces, timeTrace, fs, mean_bpm, num_heatmaps):
    if cf.DEBUG == 1:
        pulseTraces, timeTrace = (
            np.load(os.path.join(res_path,"npy_files/pulseTraces.npy")), 
            np.load(os.path.join(res_path,"npy_files/timeTrace.npy"))) 
        
    total_time = np.max(timeTrace) - np.min(timeTrace)
    time_per_heatmap = total_time / num_heatmaps            # Divise le temps total par le nombre de heatmaps pour avoir le temps par heatmaps
    
    for i in range(num_heatmaps):    # Calcul une carte par nombre de heatmaps
        start_time = i * time_per_heatmap
        end_time = start_time + time_per_heatmap
        
        mask = (timeTrace >= start_time) & (timeTrace < end_time) # Réalise un masque pour selectionné que le temps courant pour la heatmaps
        current_pulseTraces = pulseTraces[:, :, mask]
        current_timeTrace = timeTrace[mask]

        bpm_map = calculate_bpm_map_opti(current_pulseTraces,current_timeTrace, fs) # Fonction de calcul de la carte du MAE du BPM
        mae_map = np.abs(mean_bpm - bpm_map)
              
        create_mae_heatmap(mae_map, os.path.join(res_path, "heatmaps", f"MAE_MAP_{i}.png")) # Création de la carte.

       
def calculate_bpm_map(current_pulseTraces,current_timeTrace, fs):
    m, n, _ = current_pulseTraces.shape
    bpm_map = np.zeros((m, n))
    
    #HR_window_size = 5#math.floor(current_timeTrace[-1])  # <- À définir taille des fenêttres
    for i in tqdm(range(m), desc='MAE Heatmap'):
        for j in range(n):
            pixel_signal = current_pulseTraces[i, j]
            if np.all(pixel_signal == 0):
                bpm_map[i, j] = np.nan
            else:
                #bpm_values_FFT, _ = FFT(pixel_signal, current_timeTrace, 5, 0.33, 0.7, 4, fs) # On appel la fonction FFT présent dans le code getHR pour pouvoir calculé le BPM du pixel en court.
                #bpm_map[i, j] = np.mean(bpm_values_FFT) # On fait la moyenne de toutes les petites fenêtre 

                bpm_map[i, j] = BPM_FFT_simple(pixel_signal,fs,0)

    
    return bpm_map

def create_mae_heatmap(mae_map, buf):
    cmap = cm.get_cmap('jet').reversed()   
    
    fig, ax = plt.subplots()
    im = ax.imshow(mae_map, cmap=cmap, origin='upper', aspect='equal',vmax=40) # <- À définir vmax
    # Ajustement des axes pour ajouter un peu d'espace
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_xticks([])   # Supprime les graduations de l'axe X
    ax.set_yticks([])   # Supprime les graduations de l'axe Y
    ax.margins()        # Ajoute des marges autour de l'image
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05 ,dpi=300) # Sauvegardez l'image dans le tampon au lieu d'un fichier
    plt.close()


def calculate_heatmap(current_pulseTrace, fs):
    """
    Compute bpm_map
    Args:
        current_pulseTrace: ??
        fs : ??
    return:
        np.nan or np.float32
    """
    pixel_signal = current_pulseTrace
    if np.all(pixel_signal == 0):
        return np.nan
    else:
        return BPM_FFT_simple(pixel_signal,fs,0)


def calculate_bpm_map_opti(current_pulseTraces,current_timeTrace, fs):
    """
    Calcul la bpm heatmap avec joblib
    """
    m, n, _ = current_pulseTraces.shape

    results = Parallel(n_jobs=os.cpu_count())(                               # Declare autant the Job que de processeur
        delayed(lambda i,j : calculate_heatmap(current_pulseTraces[i,j], fs))(i,j)                  
        for i in tqdm(range(m), desc="MAE heatmap opti")
        for j in range(n)
    )

    bpm_map = np.array(results).reshape(m,n)
    return bpm_map

def test_bpm_equality(bpm_map_expected, bpm_map_opti, m,n):
    for i in tqdm(range(m), desc="Testing bpm_map equality"):
        for j in range(n):
            assert(bpm_map_expected[i,j]==bpm_mbpm_map_optiap2[i,j]), f"bpm_mp not equal at {i},{j} : expected {bpm_map_expected[i,j]}, got {bpm_map_opti[i,j]}"
