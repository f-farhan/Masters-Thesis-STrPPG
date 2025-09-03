"""
Description : Code de PLOT.
Permet d'afficher une figure comparant le signal de l'oxymètre et le signal de référence. Les différents BPM sont également affiché.

Un autre plot permet d'afficher les signaux bruts et le signal final correspondant pour x pixels, choisie en paramètre dans le MAIN

 
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from getHR import BPM_FFT_simple

def getPLOT(res_path, gtTime, gtTrace, gtHR,x,y,fs,mean_bpm_gtTrace,mean_bpm_sensor,mean_bpm,selected_pixels):


# ============ Signaux get rPPG ============
    # ------------ Chargements des signaux ------------
    crtTraces = np.load(os.path.join(res_path,"npy_files/crtTraces.npy"))
    pulseTraces = np.load(os.path.join(res_path,"npy_files/pulseTraces.npy"))
    timeTrace = np.load(os.path.join(res_path,"npy_files/timeTrace.npy"))
    refTraces = np.load(os.path.join(res_path,"npy_files/rppg_ref.npy"))
       
         
    # Initialiser une liste pour stocker les traces de pulse pour chaque pixel sélectionné
    all_pulseTraces = []
    all_crtTraces = []

    # Itérer sur chaque pixel sélectionné pour extraire les traces de pulse correspondantes
    for x, y in selected_pixels:
        pulseTrace = pulseTraces[y, x, :]  # [m,n,:]
        crtTrace = crtTraces[y, x, :]
        all_pulseTraces.append(pulseTrace)
        all_crtTraces.append(crtTrace)
    
    # =================== Visualisation des signaux ===================
          
    # Calculer le nombre de lignes nécessaires pour tous les pixels sélectionnés
    num_plots = len(selected_pixels)

    # Créer une figure avec un nombre dynamique de sous-tracés
    if num_plots == 1:
        fig, axs = plt.subplots(num_plots, 2, figsize=(12, 3))
        axs = np.array([axs])  # Transformer axs en tableau numpy
    else:
        fig, axs = plt.subplots(num_plots, 2, figsize=(12, num_plots * 3))

    fig.suptitle('Visualisation des signaux temporels')  # Ajustement de la position du titre

    # Tracer chaque signal sur son propre sous-tracé
    for idx, (pulseTrace, crtTrace) in enumerate(zip(all_pulseTraces, all_crtTraces)):
        axs[idx, 0].plot(timeTrace, crtTrace[0], label=f'R', color='red')
        axs[idx, 0].plot(timeTrace, crtTrace[1], label=f'G', color='green')
        axs[idx, 0].plot(timeTrace, crtTrace[2], label=f'B', color='blue')
        axs[idx, 0].set_ylabel('Amplitude')
        axs[idx, 0].set_title(f'Pixel {idx+1}')
        axs[idx, 0].legend()

        axs[idx, 1].plot(timeTrace, pulseTrace, label=f'PulseTrace')
        axs[idx, 1].set_ylabel('Amplitude')
        axs[idx, 1].set_title(f'Pixel {idx+1}')
        axs[idx, 1].legend()

    # Définir le label de l'axe x sur le dernier sous-tracé
    axs[-1, 0].set_xlabel('Temps (s)')
    axs[-1, 1].set_xlabel('Temps (s)')

    # Afficher le graphique
    plt.tight_layout()

# ============ Deuxième figure Signaux BPM ============ 
    bpm_pixel = BPM_FFT_simple(all_pulseTraces[0],fs,1)
        # Afficher le signal PPG
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))  # 1 ligne, 2 colonnes

    # Plot the first two subplots as before
    axs[0].plot(gtTime, gtTrace)
    axs[0].set_title('PPG trace Sensor')
    axs[0].set_xlabel('Temps (s)')
    axs[0].set_ylabel('Amplitude')

    axs[1].plot(timeTrace, refTraces, color='C1')
    axs[1].set_title('Signal de référence')
    axs[1].set_xlabel('Temps (s)')
    axs[1].set_ylabel('Amplitude')

    # Utiliser fig.text pour ajouter du texte en dessous des subplots
    text_str = (f"BPM Moyen GtTrace: {mean_bpm_gtTrace:.5f}"
                f"       BPM Moyen Sensor: {mean_bpm_sensor:.5f}"
                f"       BPM Moyen RefTrace: {mean_bpm:.5f}"
                f"       BPM Moyen du pixel: {bpm_pixel:.5f}")
    fig.text(0.5, 0.02, text_str, ha='center', fontsize=13, fontweight='bold')

    plt.tight_layout(pad=3.0)  # Ajuster les espacements pour inclure le texte en bas





    """
    crtTrace = crtTraces[y, x,:,:] #car [m,n,:]
    filterTrace = filterTraces[y, x,:, :] #car [m,n,:]
    pulseTrace = pulseTraces[y, x, :] #car [m,n,:]

# ============ Première FIGURE  ============
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18,8))
    fig.suptitle('Get rPPG', y=0.92, fontsize=16)
    fig.suptitle('Get rPPG - Coordonnées: x={}, y={}'.format(x, y), y=1, fontsize=16)
    fig.tight_layout()  # Ajustement de l'espacement entre les sous-graphiques
    
    # ------------ Trace RGB ------------   
    axs[0, 0].plot(timeTrace, crtTrace[0, :], color='r', linewidth=1)
    axs[0, 0].plot(timeTrace, crtTrace[1, :], color='g', linewidth=1)
    axs[0, 0].plot(timeTrace, crtTrace[2, :], color='b', linewidth=1)
    axs[0, 0].set_title('Full RGB traces')
    axs[0, 0].set_xlabel('Temps (s)')
    axs[0, 0].set_ylabel('Amplitude')           
    
    # ------------ Filtre butterworth ------------
    axs[0, 1].plot(timeTrace, filterTrace[0, :], color='r', linewidth=1)
    axs[0, 1].plot(timeTrace, filterTrace[1, :], color='g', linewidth=1)
    axs[0, 1].plot(timeTrace, filterTrace[2, :], color='b', linewidth=1)
    axs[0, 1].set_title('Filtre Butterworth')
    axs[0, 1].set_ylabel('Amplitude ')  
    axs[0, 1].set_xlabel('Temps (s)')

    axs[1, 0].set_axis_off()
    axs[1, 1].set_axis_off()
    
    # Ajustement de l'espacement entre les sous-graphiques
    fig.tight_layout()
    
    # ------------ Signal Après les Méthodes ------------
    # 3ème plot sur toute la longeur 
    ax_combined1 = fig.add_subplot(3, 2, (5, 6))

    ax_combined1.plot(timeTrace, pulseTrace/np.max(pulseTrace), color='b', linewidth=1) #Signal Méthode choisi
    #ax_combined1.plot(gtTime, gtTrace/np.max(gtTrace), 'k') #Signal PPG
    ax_combined1.set_xlim([10, 25])  
    ax_combined1.set_title('Comparaison des signaux après la combinaison de chanel')
    ax_combined1.set_xlabel('Temps (s)')
    ax_combined1.set_ylabel('Amplitude (normalized)')    





    # Afficher le signal PPG 
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 8), gridspec_kw={'height_ratios': [1, 2]})

    # Plot the first two subplots as before
    axs[0, 0].plot(gtTime, gtTrace) 
    axs[0, 0].set_title('PPG trace Sensor')
    axs[0, 0].set_xlabel('Temps (s)')
    axs[0, 0].set_ylabel('Amplitude')


    axs[0, 1].plot(timeTrace, pulseTrace,color='C1')
    axs[0, 1].set_title('rPPG trace')
    axs[0, 1].set_xlabel('Temps (s)')
    axs[0, 1].set_ylabel('Amplitude')
    

    axs[1, 0].set_axis_off()
    axs[1, 1].set_axis_off()

    # 3ème Plot taille complet
    ax_combined2 = fig.add_subplot(2, 2, (3, 4))
    p1, = ax_combined2.plot(start_times_PPG, bpm_values_PPG, 's-', label='PPG')
    p2, = ax_combined2.plot(start_times_rPPG, bpm_values_rPPG, 's-', label='rPPG')
    p3, = ax_combined2.plot(gtTime, gtHR, '--', label='Sensor')
    ax_combined2.set_title('Comparaison du BPM')
    ax_combined2.set_xlabel('Temps (s)')
    ax_combined2.set_ylabel('BPM')
    ax_combined2.legend(handles=[p1, p2, p3], labels=['PPG', 'rPPG', 'Sensor'])
    """