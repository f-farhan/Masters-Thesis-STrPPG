"""
Description : Code pour générer la carte d'amplitude moyennée, avec la méthode fréquentielle FFT.
On analyse l'énergie présente dans la bande de fréquence autour du BPM moyen, ce qui nous donne une valeur pour chaque pixel.  

Définition manuelle de la vmax des cartes (Normalisation).
Bande de fréquence arbitraire, à définir selon les cas souhaités.
"""
import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from joblib import Parallel, delayed
from config import cf
# ============ Fonction Principal ============ 
def amplitude_heatmap_FRQ(res_path, pulseTraces, timeTrace,mean_bpm, lowF, upF, refTraces, num_heatmaps):
    
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
        current_refTraces = refTraces[mask]


        amp_map = calculate_amplitude_FRQ_opti(current_pulseTraces,timeTrace,mean_bpm, lowF, upF,current_refTraces)
        #amp_map_origin = calculate_amplitude_FRQ(current_pulseTraces,timeTrace,mean_bpm, lowF, upF,current_refTraces)
        #test_amp_freq(amp_map_origin, amp_map, current_pulseTraces.shape)
        cmap = cm.get_cmap('jet')  # Choix de la colormap jet
        
        fig, ax = plt.subplots()
        im = ax.imshow(amp_map, cmap=cmap, origin='upper', aspect='equal', vmax=1500) #<- À définir vmax
        # Ajustement des axes pour ajouter un peu d'espace
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_xticks([])   # Supprime les graduations de l'axe X
        ax.set_yticks([])   # Supprime les graduations de l'axe Y
        ax.margins()        # Ajoute des marges autour de l'image
        plt.tight_layout()
        plt.savefig(os.path.join(res_path, "heatmaps", f'{"AMPL_MAP_FRQ"}_{i}.png'), bbox_inches='tight',  format='png',pad_inches=0.05,dpi=300)
        plt.close()

        
# ============ Fonction de calcul de l'amplitude ============
def calculate_amplitude_FRQ(current_pulseTraces,timeTrace,mean_bpm, lowF, upF,current_refTraces):
    m, n, t = current_pulseTraces.shape
    amp_map = np.zeros((m, n))

    for i in tqdm(range(m), desc='FRQ'):
        for j in range(n):
            pixel_signal = np.nan_to_num(current_pulseTraces[i, j, :]) #Pour chaque pixel # Remplacer les infs et NaNs par des valeurs valides
            
            # ============ Methode Fréquenciel Energie ============

            window = np.hanning(len(pixel_signal))  # Crée une fenêtre de Hanning de la même taille que le signal
            windowed_signal = pixel_signal * window 
            pulseTrace_FFT = np.fft.fft(windowed_signal)
            frq = np.fft.fftfreq(len(pixel_signal), timeTrace[1] - timeTrace[0])                        # Fréquences correspondantes
            lowF = (mean_bpm - (mean_bpm * 0.1))/60                                                    # Bande de fréquence autour de la fréquence du BPM
            upF = (mean_bpm + (mean_bpm * 0.1))/60
            
            indices = np.where(((frq >= lowF) & (frq <= upF)) | ((frq >= 2 * lowF) & (frq <= 2 * upF))) # Récupération des indices correspondant aux fréquences d'intérêt, y compris la deuxième harmonique
            amp_map[i, j]= np.sum(np.abs(pulseTrace_FFT[indices]) ** 2)                                       # Calcul de l'énergie 
           
            #else: 
                    #energy = np.nan  # Si inférieur alors energy = null   
    
    return amp_map

def freq_opti(pixel_signal, timeTrace,mean_bpm, lowF, upF,current_refTraces):
    pixel_signal = np.nan_to_num(pixel_signal)
    window = np.hanning(len(pixel_signal))  # Crée une fenêtre de Hanning de la même taille que le signal
    windowed_signal = pixel_signal * window 
    pulseTrace_FFT = np.fft.fft(windowed_signal)
    frq = np.fft.fftfreq(len(pixel_signal), timeTrace[1] - timeTrace[0])                        # Fréquences correspondantes
    lowF = (mean_bpm - (mean_bpm * 0.1))/60                                                    # Bande de fréquence autour de la fréquence du BPM
    upF = (mean_bpm + (mean_bpm * 0.1))/60
    
    indices = np.where(((frq >= lowF) & (frq <= upF)) | ((frq >= 2 * lowF) & (frq <= 2 * upF))) # Récupération des indices correspondant aux fréquences d'intérêt, y compris la deuxième harmonique
    return np.sum(np.abs(pulseTrace_FFT[indices]) ** 2)                                       # Calcul de l'énergie 
    
    
def calculate_amplitude_FRQ_opti(current_pulseTraces,timeTrace,mean_bpm, lowF, upF,current_refTraces):
    m, n, t = current_pulseTraces.shape
    amp_map = np.zeros((m, n))

    
    results = Parallel(n_jobs=os.cpu_count())(
        delayed(freq_opti)(current_pulseTraces[i,j],timeTrace,mean_bpm, lowF, upF,current_refTraces)
        for i in tqdm(range(m), desc="FRQ opti")
        for j in range(n)
    )    

    amp_map = np.array(results).reshape((m,n))
    return amp_map


def test_amp_freq(freq_origin, freq_opti, shape):
    for i in range(shape[0]):
        for j in range(shape[1]):
            assert(freq_origin[i,j]== freq_opti[i,j]), f"Freq map not equal at index {i},{j} : {freq_origin[i,j]} != {freq_opti[i,j]}"
