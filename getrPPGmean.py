"""
Description : Un des codes les plus importants. 
Il permet de traiter les données pour obtenir le signal PPG moyenné pour avoir le signal de référence.

Des zones de code sont commentées, elles la pour réalisé quelques test de méthodes

 
Plusieurs filtres peuvent être utilisés, tels que passe-bas, passe-bandes, Butterworth, FIR. 
Ces filtres doivent impérativement être appliqués après la combinaison des canaux pour ne pas perturber certaines méthodes. 

"""

import os
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt,find_peaks
from scipy.fft import fft, fftfreq

from joblib import Parallel, delayed
from tqdm import tqdm
from utils import detrend,bandpass_filter,trouver_pics,smooth_filter
from config import cf

def getrPPGmean(ROI_REF,res_path,fs,lowF,upF,filtOrder):
    
    # Butterworth filter parameters
    nyq = 0.5 * fs  
    low = lowF / nyq 
    high = upF / nyq 
    b, a = butter(filtOrder, [low, high], btype='band') 

    timeTrace = np.linspace(0, (ROI_REF.shape[3] - 1) / fs, ROI_REF.shape[3])
    winLength = round(cf.WINLENGTHSEC * fs) 

    # Spatial averaging to get RGB temporal traces
    rgb_ref_traces = np.mean(np.mean(ROI_REF, keepdims = True, axis = 0, dtype=np.float32), axis=1, keepdims = True, dtype=np.float32)
    # rPPG reference signal estimation from RGB temporal traces
    rppg_ref = process(cf,rgb_ref_traces,timeTrace,winLength,lowF,upF,a,b,fs) 
         
    if cf.DEBUG == 1:
        # save rppg reference signal as an image 
        fig = plt.figure(figsize=(12, 4))                                   
        plt.subplots_adjust(left=0.05, right=0.99, bottom=0.08, top=0.97)   
        plt.plot(timeTrace,rppg_ref)
        plt.savefig(os.path.join(res_path,"rppg_ref.png")) 

    # save binary files
    np.save(os.path.join(res_path,"npy_files","rppg_ref.npy"), rppg_ref) 
    np.save(os.path.join(res_path,"npy_files","timeTrace.npy"), timeTrace)    

    return rppg_ref, timeTrace

# =============== Traitement du signal moyenné ===============
def process(cf, rgb_ref_traces,timeTrace,winLength,lowF,upF,a,b,fs):

    crtTrace = np.squeeze(rgb_ref_traces[0,0,0:3,:])
    traceSize = crtTrace.shape[1]  
    
    filterTrace = np.copy(crtTrace)
    # ------------ Traitement de signal ------------ 
    if cf.DETREND == 1:
        filterTrace[k,:] = detrend(filterTrace[k,:],500) 
        
        for k in range(3): # Multiplication de la fenêtre de Hanning
            hann_window = np.hanning(filterTrace.shape[1])

        filterTrace[k, :] *= hann_window
    
    if cf.NORMALISATION == 1:
        tmpo = np.zeros(filterTrace.shape)
        for t in range(filterTrace.shape[1] - winLength + 1):
            C = filterTrace[:, t:t + winLength - 1]
            tmpo[:, t:t + winLength - 1] = tmpo[:, t:t+winLength-1] + np.linalg.inv(np.diag(np.mean(C, axis=1))) @ C - 1
        
        filterTrace = tmpo  
        del tmpo  
                                        
    # ------------ CHANNEL COMBINATION ------------
    if cf.METHOD == 'Green':              
        pulseTrace = filterTrace[1,:]
    elif cf.METHOD == 'G-R':
        pulseTrace = filterTrace[1,:] - filterTrace[0,:]
    elif cf.METHOD == 'Chrom':
        pulseTrace = np.zeros(traceSize)
        for t in range(0, filterTrace.shape[1] - winLength + 1):
            C_window = filterTrace[:, t:t+winLength-1]
            mean_C = np.mean(C_window, axis=1) # Calcul de la moyenne de chaque ligne de la matrice C
            diag_mean_C = np.diag(mean_C) # Création de la matrice diagonale avec les moyennes calculées
            inv_diag_mean_C = np.linalg.inv(diag_mean_C) # Calcul de l'inverse de la matrice diagonale
            Cn = np.dot(inv_diag_mean_C, C_window) # Calcul de Cn en appliquant la normalisation temporelle
            Rf = Cn[0]
            Gf = Cn[1]
            Bf = Cn[2]
            Xf = 3*Rf - 2*Gf
            Yf = 1.5*Rf + Gf - 1.5*Bf
            alpha = 1
            #alpha = np.std(Xf)/np.std(Yf)
            S = Xf - alpha*Yf
            pulseTrace[t:t+winLength-1] += (S - np.mean(S))/np.std(S) 
    elif cf.METHOD == 'POS':
        pulseTrace = np.zeros(traceSize)
        for t in range(0, filterTrace.shape[1] - winLength + 1):     
            C_window = filterTrace[:, t:t+winLength-1]
            mean_C = np.mean(C_window, axis=1)              # Calcul de la moyenne de chaque ligne de la matrice C
            diag_mean_C = np.diag(mean_C)                   # Création de la matrice diagonale avec les moyennes calculées
            inv_diag_mean_C = np.linalg.inv(diag_mean_C)    # Calcul de l'inverse de la matrice diagonale
            Cn = np.dot(inv_diag_mean_C, C_window)          # Calcul de Cn en appliquant la normalisation temporelle
            S = np.array([[0, 1, -1],[-2, 1, 1]])
            S = np.dot(S,Cn)
            P = np.dot([1, 1], S)
            #pulseTrace[t:t+winLength-1] += (P - np.mean(P))/np.std(P)  # overlap add
            pulseTrace[t:t+winLength-1] += P

    elif cf.METHOD == 'POS2': #METHODE D'UNE PUBLICATION  #Xs = g(t) - b(t) & Ys = -2r(t) + g(t) + b(t)  
        Xs = filterTrace[1,:] - filterTrace[2,:]
        Ys = -2*filterTrace[0,:] + filterTrace[1,:] + filterTrace[2,:]
        pulseTrace = Xs + (1)* Ys
        #pulseTrace = Xs + (np.std(Xs) / np.std(Ys))* Ys

    # ------------ FILTRAGE  -----------------
    pulseTrace = filtfilt(b, a, pulseTrace) #Filtre Passe Bande ou Passe Bas

    if cf.SHORTFRQ == 1:
        for k in range(3):
            ref_freq, h_freq = trouver_pics(pulseTrace[k, :], timeTrace)                                               # Trouves les fréquences 1ère et 2ème harmoniques
            if ref_freq is not None and h_freq is not None:                                                             # Créer les bandes de fréquences
                bande1 = (ref_freq - 0.1, ref_freq + 0.1)                                                               # Créer les bandes de fréquences 1er harmonique
                bande2 = (h_freq - 0.1, h_freq + 0.1)                                                                   # Créer les bandes de fréquences 2ème harmonique  
                frequencies = [bande1, bande2]                                                                          # Filtrer le signal dans les bandes de fréquences spécifiées
                pulseTrace[k, :] = bandpass_filter(pulseTrace[k, :], frequencies, 1 / (timeTrace[1] - timeTrace[0]))    # Filtre Passe Bande FFT          

    return pulseTrace

def get_rppg_mean(source, RppgD, roi):
# =============== Calcul coefs filtre butterworth ===============
    nyq = 0.5 * RppgD.fs                                              # Calcul de la fréquence de Nyquist
    low = RppgD.lowF / nyq                                            # Normalisation des fréquences de coupure
    high = RppgD.upF / nyq 
    b, a = butter(RppgD.filter_order, [low, high], btype='band')         # Calcul des coefficients du filtre Butterwort
    #b, a = butter(filtOrder, high, btype='low', analog=False)  # Passe Bas

# =============== Création des Variables ===============
    timeTrace = np.linspace(0, (roi.nb_frames - 1) / RppgD.fs, roi.nb_frames)
    winLength = round(cf.WINLENGTHSEC * RppgD.fs) 

# =============== Traitement ===============
    roi_mean = np.mean(np.mean(roi.frames, keepdims = True, axis = 0, dtype=np.float32), axis=1, keepdims = True, dtype=np.float32) # Moyennage de tout les pixels 
    refTrace_new = process(cf,roi_mean, timeTrace,winLength,RppgD.lowF,RppgD.upF,a,b,RppgD.fs)                   # Traitement
    return refTrace_new, timeTrace

def get_green_channel(roi_mean,a,b):
    crtTrace  = np.squeeze(roi_mean[0:3, :])
    traceSize = crtTrace.shape[1]
    filterTrace = np.copy(crtTrace)
    pulseTrace = filterTrace[1,:] # green channel
    pulseTrace = filtfilt(b, a, pulseTrace) #Filtre Passe Bande ou Passe Bas
    return pulseTrace

def process_opti(roi_mean, a, b):
    """
    NE PAS UTILISER, moins performante que l'original
    """
    results = Parallel(n_jobs=os.cpu_count())(
        delayed(get_green_channel)(roi_mean[m,n, :, :], a,b)
        for m in tqdm(range(roi_mean.shape[0]), desc="RppgMean")
        for n in range(roi_mean.shape[1])
    )
    pulseTrace = np.array(results)
    return pulseTrace

def test_rppg_mean(ref_origin, ref_opti):
    for i in range(ref_origin.shape[0]):
        for j in range(ref_origin.shape[1]):
            for k in range(ref_origin.shape[2]):
                for l in range(ref_origin.shape[3]):
                    assert(ref_opti[i,j,k,l]== ref_origin[i,j,k,l]), f"Rppg Mean not equal at {i},{j},{k},{l}, expected {ref_origin[i,j,k,l]} got {ref_opti[i,j,k,l]}"


    
