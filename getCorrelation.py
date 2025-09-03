"""
Description : Fonction pour générer la carte de corrélation entre le signal de chaque pixel (pulseTraces) et le signal de référence (refTraces).

On multiplie notre signal par pixel par un signal de référence et, en calculant la racine carrée des parties réelle et imaginaire, on obtient l'amplitude désirée.
Pour la corrélation, les valeurs vont de -1 à 1, avec 1 indiquant une corrélation totale, 0 aucune corrélation, et -1 une corrélation inverse.
Carte P = probabilité de trouver le résultat si le coefficient de corrélation était nul.

Le calcul de la carte de corrélation est plutôt long.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') #newly added
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable

from joblib import Parallel, delayed
from config import cf
# ============ Fonction Principal ============
def correlation_heatmap(res_path, pulseTraces, timeTrace, refTraces, num_heatmaps):
    if cf.DEBUG == 1:
       pulseTraces, timeTrace, refTraces = (
            np.load(os.path.join(res_path,"npy_files/pulseTraces.npy")), 
            np.load(os.path.join(res_path,"npy_files/timeTrace.npy")), 
            np.load(os.path.join(res_path,"npy_files/rppg_ref.npy"))) 

    total_time = np.max(timeTrace) - np.min(timeTrace)
    time_per_heatmap = total_time / num_heatmaps    # Divise le temps total par le nombre de heatmaps pour avoir le temps par heatmaps

    for i in range(num_heatmaps): # Calcul une carte par nombre de heatmaps
        start_time = i * time_per_heatmap
        end_time = start_time + time_per_heatmap

        
        mask = (timeTrace >= start_time) & (timeTrace < end_time) # Réalise un masque pour selectionné que le temps courant pour la heatmaps
        current_pulseTraces = pulseTraces[:, :, mask]
        current_refTraces = refTraces[mask]
        
        corr, opti_p = calculate_map_opti(current_pulseTraces, current_refTraces)
        coefCorr, p = calculate_map(current_pulseTraces, current_refTraces) # Fonction de calcul de la carte de corrélation 
        # test_get_correlation(coefCorr, p, corr, opti_p,current_pulseTraces.shape )
        create_heatmap(coefCorr, os.path.join(res_path, "heatmaps", f"CORR_MAP_{i}.png"))  # Corrélation MAP : Echelle : -1 à 1 #changed this line from coefCorr
        create_heatmap(p, os.path.join(res_path, "heatmaps", f'P_MAP_{i}.png'))           # P MAP : Echelle : 0.0 à 0.10 --> cmap = cm.get_cmap('jet').reversed() 
        


        # ============ Fonction Calcul ============
def calculate_map(ppg_signal, current_refTraces):
    m, n, _ = ppg_signal.shape
    coefCorr = np.zeros((m, n))
    p_value = np.zeros((m, n))
    
    for i in tqdm(range(m),desc='CORR Heatmap'):  # Ajout de tqdm pour la barre de progression
        for j in range(n):
            i, j, corr, p = calculate_pixel(i, j, ppg_signal, current_refTraces)
            
            coefCorr[i, j] = corr
            p_value[i, j] = p
    
    return coefCorr, p_value


# ============ Fonction de calcul de corrélation ============
def calculate_pixel(i, j, ppg_signal, current_refTraces):
    pixel_signal = np.nan_to_num(ppg_signal[i, j])
    corr, p = pearsonr(current_refTraces, pixel_signal) #Utilisation de la fonction PEARSON R pour le calcul de la corrélation entre deux signaux. 

    return i, j, corr, p


# ============ Fonction Création Heatmap ============
def create_heatmap(corr_map, buf):
    cmap = cm.get_cmap('jet')
   
    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(corr_map), cmap=cmap, origin='upper', aspect='equal', vmin=0, vmax=1)  # Choix de l'échelle
    # Ajustement des axes pour ajouter un peu d'espace
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_xticks([])   # Supprime les graduations de l'axe X
    ax.set_yticks([])   # Supprime les graduations de l'axe Y
    ax.margins()        # Ajoute des marges autour de l'image
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05,dpi=300)  # Sauvegardez l'image dans le tampon au lieu d'un fichier
    plt.close()

def calculate_pixel_opti(ppg_signal, current_trace):
    pixel_signal = np.nan_to_num(ppg_signal)
    corr, p = pearsonr(current_trace, pixel_signal)
    return corr, p

def calculate_map_opti(ppg_signal, current_trace):
    """
    Calcul de la carte des correlations
    """

    m, n, _ = ppg_signal.shape
    coefCorr = np.zeros((m, n))
    p_value = np.zeros((m, n))

    results = Parallel(n_jobs=os.cpu_count())(
        delayed(calculate_pixel_opti)(ppg_signal[i,j], current_trace)
        for i in tqdm(range(m), desc="CORR heatmap opti")
        for j in range(n)
    )
    corr, p = zip(*results)
    corr    = np.array(corr).reshape((m,n))
    p       = np.array(p).reshape((m,n))

    return corr, p

def test_get_correlation(o_corr, o_p, opti_cor, opti_p, shape):

    for i in range(shape[0]):
        for j in range(shape[1]):
            assert(o_corr[i, j]==opti_cor[i,j]), f"correlation not egal at {i} {j}: {o_corr[i,j]} != {opti_cor[i,j]}"
            assert(o_p[i, j]==opti_p[i,j]), f"correlation not egal at {i} {j}: {o_p[i,j]} != {opti_p[i,j]}"
