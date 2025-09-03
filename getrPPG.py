"""
Description : Un des codes les plus importants. 
Il permet de traiter les données pour obtenir le signal PPG par pixel. Pour des raisons de test, trois tenseurs sont enregistrés : 
Premier tenseur (crtTraces) : enregistre les signaux bruts RGB par pixel.
Deuxième tenseur (filterTraces) : enregistre les signaux filtrés par pixel —> ! Seulement si un filtrage à été effectué avant la combinaison de canaux, sinon le tenseur correspond au signaux Brutes 
Troisième tenseur (pulseTraces) : enregistre les signaux finaux par pixel.

Des zones de code sont commentées, elles sont utilisées principalement pour éliminer le fond de l'acquisition, via des seuils de couleur ou des arrangements des canaux RGB.

Deux types de traitement sont présents : l'un séquentiel, et l'autre avec multiprocessing. 
Le multiprocessing permet de gagner du temps lorsque l'on utilise les méthodes POS ou Chrom qui demandent plus de temps de traitement


 
La taille du ROI ne doit pas être excessive, car sinon la taille des tenseurs crtTraces et filterTraces pose problème et dépasse la capacité mémoire allouée.
Le processus peut être très lourd, occupant plus de 15-20 Go de mémoire.

Plusieurs filtres peuvent être utilisés, tels que passe-bas, passe-bandes, Butterworth, FIR. 
Ces filtres doivent impérativement être appliqués après la combinaison des canaux pour ne pas perturber certaines méthodes. 

"""

import os
import numpy as np

import multiprocessing
from tqdm import tqdm
from scipy.signal import butter, filtfilt,find_peaks
from scipy.fft import fft, fftfreq
from utils import detrend,bandpass_filter,trouver_pics,smooth_filter

import time
from joblib import Parallel, delayed
from config import cf

def get_rppg(roi, res_path, rppgd):
    getrPPG(roi, res_path, rppgd.fs, rppgd.lowF, rppgd.upF, rppgd.filter_order)

# =============== Fonction Principal ===============
def getrPPG(ROI,res_path,fs,lowF,upF,filtOrder):

    # Butterworth filter parameters
    nyq = 0.5 * fs  
    low = lowF / nyq 
    high = upF / nyq 
    b, a = butter(filtOrder, [low, high], btype='band') 
    
    SEQUENTIAL = cf.METHOD=="Chrom"

    winLength = round(cf.WINLENGTHSEC * fs) # get window length in frames
    crtTraces_new = np.zeros((ROI.shape[0], ROI.shape[1],3, ROI.shape[3]))
    #filterTraces_new = np.zeros((ROI.shape[0], ROI.shape[1],3, ROI.shape[3]))
    pulseTraces_new = np.zeros((ROI.shape[0], ROI.shape[1], ROI.shape[3])) 
    timeTrace = np.linspace(0, (ROI.shape[3] - 1) / fs, ROI.shape[3])

# =============== Traitement Sequentiel ===============
    if SEQUENTIAL:
        print("SEQUENTIAL")
        # crtTraces_new,pulseTraces_new = sequential_process(cf,ROI,winLength,timeTrace, a,b)
        crtTraces_new,pulseTraces_new = sequential_process_opti(ROI, a,b)
        
# =============== Traitement Multiprocess ===============       
    else: 
        # ------------ Initialisation ------------
        print("MULTIPROCESS")
        num_processes = multiprocessing.cpu_count()                 # Nombre de processeurs disponibles
        num_elements_n = ROI.shape[1]                               # Nombre total d'éléments dans ROI
        base_elements_per_task_n = num_elements_n // num_processes  # Division entière
        extra_elements_n = num_elements_n % num_processes           # Reste de la division

        # ------------ Création des tâches ------------
        tasks = []                                                  # Liste pour stocker les tâches à exécuter
        n_start = 0                                                 # Indice de début pour chaque tâche
        for i in range(num_processes):
            elements_per_task_n = base_elements_per_task_n + (1 if i < extra_elements_n else 0)
            n_end = n_start + elements_per_task_n
            tasks.append((ROI,winLength,timeTrace,a,b,n_start, n_end))    # Ajout de la tâche avec ses paramètres dans la liste des tâches
            n_start = n_end                                                                             # Préparer n_start pour la prochaine itération
            
        # ------------ Exécution en parallèle ------------
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(multi_process, tasks)        #Exécution des tâches en parallèle et récupération des résultats
        
        # ------------ Collecte et tri des résultats ------------
        sorted_results = sorted(results, key=lambda x: x[0])    # Tri des résultats par indice de début
         
        # ------------ Collecte des résultats ------------
        for result in sorted_results:
            n_start, crtTraces, pulseTraces = result
            n_end = n_start + crtTraces.shape[1] # Calcul de l'indice de fin pour chaque tâche

            # Mise à jour des matrices avec les résultats de chaque tâche
            crtTraces_new[:, n_start:n_end, :] = crtTraces
            pulseTraces_new[:, n_start:n_end, :] = pulseTraces

# =============== Enregistrement des Variables ===============         
    if cf.DEBUG == 1:  
        np.save(os.path.join(res_path,"npy_files","crtTraces.npy"), crtTraces_new)        # Signal Trace RGB
        np.save(os.path.join(res_path,"npy_files","pulseTraces.npy"), pulseTraces_new)    # Signal Après Méthode 
        np.save(os.path.join(res_path,"npy_files","timeTrace.npy"), timeTrace)            # Signal Temps       # YBTODO : also saved in getrPPGmean, duplicate ? 

    return pulseTraces_new

# =============== Fonction Traitement Multiprocess ===============
def multi_process(ROI, winLength,timeTrace,a,b,n_start, n_end):
   
    crtTraces = np.zeros((ROI.shape[0], n_end - n_start,3, ROI.shape[3]))  
    pulseTraces = np.zeros((ROI.shape[0], n_end - n_start, ROI.shape[3]))  

    # Traiter chaque tranche de `ROI`
    for n in range(n_end - n_start):
        n_abs = n + n_start
        print("%i over %i" % (n_abs+1, n_end))
        for m in range(ROI.shape[0]):
            crtTrace = np.squeeze(ROI[m, n_abs, 0:3, :])
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
                
                filterTrace = tmpo  # Remplacement de normTrace par la matrice temporaire normalisée
                del tmpo  # Suppression de la matrice temporaire

                """ # Autre type de normalisation
                normTrace = np.copy(filterTrace)
            
                L = winLength
                tmp = np.zeros(normTrace.shape)
                Normalisation = True
                for t in range(normTrace.shape[1] - L + 1):
                    C = normTrace[:, t:t + L - 1]
                    mean_values = np.mean(C, axis=1)  # Calcul des moyennes des canaux de couleur
                    normalized_C = C - mean_values[:, np.newaxis]  # Différence par la moyenne de chaque canal
                    ac_values = np.std(normalized_C, axis=1)  # Calcul des amplitudes des composantes alternatives (AC)
                    normalized_C /= ac_values[:, np.newaxis]  # Division par l'amplitude de chaque canal
                    tmp[:, t:t + L - 1] = tmp[:, t:t + L - 1] + normalized_C  # Ajout de la différence normalisée à la matrice temporaire
                
                normTrace = tmp  # Remplacement de normTrace par la matrice temporaire normalisée
                del tmp  # Suppression de la matrice temporaire"""
                          
                        
            if cf.REMOVE_BACKGROUND == 1:     
                for k in range(3):
                    Moyenne_Couleur =  np.mean(crtTrace, axis=-1)
                    if  Moyenne_Couleur[0] <= 240 :
                        filterTrace[k,:] = filterTrace[k,:]
                    else :     
                        filterTrace[k,:] = np.zeros_like(filterTrace[k,:])   

                    if  Moyenne_Couleur[0] > 65 :
                        filterTrace[k,:] = filterTrace[k,:]
                    else :     
                        filterTrace[k,:] = np.zeros_like(filterTrace[k,:])

                    if  Moyenne_Couleur[2] > 150 :
                        if Moyenne_Couleur[0] > Moyenne_Couleur[2] :
                            filterTrace[k,:] = filterTrace[k,:] 
                        else :
                            filterTrace[k,:] = np.zeros_like(filterTrace[k,:])
                
                    
                    if Moyenne_Couleur[1] > Moyenne_Couleur[2] :
                        filterTrace[k,:] = filterTrace[k,:] 
                    else :
                        filterTrace[k,:] = np.zeros_like(filterTrace[k,:])
            
                    if Moyenne_Couleur[2] < 180 :                    
                        filterTrace[k,:] = filterTrace[k,:] 
                    else :
                        filterTrace[k,:] = np.zeros_like(filterTrace[k,:])
                    
                    
                    # CODE TEST : Pour supprimer les pixels avec des valeurs d'amplitude très excessive . 
                    fft_signal = np.fft.fft(filterTrace[k,:])
                    frequencies = np.fft.fftfreq(len(fft_signal), timeTrace[1] - timeTrace[0])
                    
                    power_spectrum = np.abs(fft_signal)**2 # Calculez le spectre de puissance
                    max_power = np.max(power_spectrum)  # Trouvez la valeur de la puissance maximale
                    
                    if max_power > 40000 : 
                        filterTrace[k,:] = np.zeros_like(filterTrace[k,:])  # Ou utilisez np.zeros_like(signal) pour mettre à 0
                    else:
                        filterTrace[k,:] = filterTrace[k,:] 


            # ------------ CHANNEL COMBINATION ------------   
            if np.all(filterTrace != 0):   
                if cf.METHOD == 'Green':              
                    pulseTrace = filterTrace[1,:]
                elif cf.METHOD == 'G-R':
                    pulseTrace = filterTrace[1,:] - filterTrace[0,:]
                elif cf.METHOD == 'Chrom':
                    pulseTrace = np.zeros(traceSize)
                    for t in range(len(filterTrace[1]) - winLength + 1):
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
                    pulseTrace = Xs + (np.std(Xs) / np.std(Ys))* Ys

             
            # ------------ FILTRAGE  -----------------
            #pulseTrace = filtfilt(b, a, pulseTrace) #Filtre Passe Bande ou Passe Bas 
            if 'pulseTrace' in locals():
                 pulseTrace = filtfilt(b, a, pulseTrace)  # Filtre Passe Bande ou Passe Bas
            else:
                pulseTrace = np.zeros(traceSize)  # fallback if pulseTrace wasn't computed  

            if cf.SHORTFRQ == 1:
                for k in range(3):
                    ref_freq, h_freq = trouver_pics(pulseTrace[k, :], timeTrace)                                               # Trouves les fréquences 1ère et 2ème harmoniques
                    if ref_freq is not None and h_freq is not None:                                                             # Créer les bandes de fréquences
                        bande1 = (ref_freq - 0.1, ref_freq + 0.1)                                                               # Créer les bandes de fréquences 1er harmonique
                        bande2 = (h_freq - 0.1, h_freq + 0.1)                                                                   # Créer les bandes de fréquences 2ème harmonique  
                        frequencies = [bande1, bande2]                                                                          # Filtrer le signal dans les bandes de fréquences spécifiées
                        pulseTrace[k, :] = bandpass_filter(pulseTrace[k, :], frequencies, 1 / (timeTrace[1] - timeTrace[0]))  # Filtre Passe Bande FFT      
            
            crtTraces[m, n,:, :] = crtTrace         
            pulseTraces[m, n, :] = pulseTrace
    
    return n_start, crtTraces, pulseTraces  

# =============== Fonction Traitement Sequentiel ===============
def sequential_process(cf, ROI, winLength,timeTrace,a,b):
    crtTraces = np.zeros((ROI.shape[0], ROI.shape[1],3, ROI.shape[3]))  
    pulseTraces = np.zeros((ROI.shape[0], ROI.shape[1], ROI.shape[3]))
    for m in tqdm(range(ROI.shape[0]), desc='Rows'): # Traiter chaque tranche de `ROI`
        for n in range(ROI.shape[1]):
            crtTrace = np.squeeze(ROI[m,n,0:3,:])
            traceSize = crtTrace.shape[1]

            filterTrace = np.copy(crtTrace)
            # ------------ Traitement de signal ------------ 
            if cf.DETREND == 1:
                for k in range(3): 
                    filterTrace[k,:] = detrend(filterTrace[k,:],500) 
                
                for k in range(3): # Multiplication de la fenêtre de Hanning
                    hann_window = np.hanning(filterTrace.shape[1])

                filterTrace[k, :] *= hann_window
            
            if cf.NORMALISATION == 1:
                tmpo = np.zeros(filterTrace.shape)
                for t in range(filterTrace.shape[1] - winLength + 1):
                    C = filterTrace[:, t:t + winLength - 1]
                    tmpo[:, t:t + winLength - 1] = tmpo[:, t:t+winLength-1] + np.linalg.inv(np.diag(np.mean(C, axis=1))) @ C - 1
                
                filterTrace = tmpo  # Remplacement de normTrace par la matrice temporaire normalisée
                del tmpo  # Suppression de la matrice temporaire

                """ # Autre type de normalisation
                normTrace = np.copy(filterTrace)
            
                L = winLength
                tmp = np.zeros(normTrace.shape)
                Normalisation = True
                for t in range(normTrace.shape[1] - L + 1):
                    C = normTrace[:, t:t + L - 1]
                    mean_values = np.mean(C, axis=1)  # Calcul des moyennes des canaux de couleur
                    normalized_C = C - mean_values[:, np.newaxis]  # Différence par la moyenne de chaque canal
                    ac_values = np.std(normalized_C, axis=1)  # Calcul des amplitudes des composantes alternatives (AC)
                    normalized_C /= ac_values[:, np.newaxis]  # Division par l'amplitude de chaque canal
                    tmp[:, t:t + L - 1] = tmp[:, t:t + L - 1] + normalized_C  # Ajout de la différence normalisée à la matrice temporaire
                
                normTrace = tmp  # Remplacement de normTrace par la matrice temporaire normalisée
                del tmp  # Suppression de la matrice temporaire"""
                          
                
            if cf.REMOVE_BACKGROUND == 1:     
                for k in range(3):
                    Moyenne_Couleur =  np.mean(filterTrace, axis=-1)
                    """
                    if  Moyenne_Couleur[0] <= 240 :
                        filterTrace[k,:] = filterTrace[k,:]
                    else :     
                        filterTrace[k,:] = np.zeros_like(filterTrace[k,:])   

                    if  Moyenne_Couleur[0] > 65 :
                        filterTrace[k,:] = filterTrace[k,:]
                    else :     
                        filterTrace[k,:] = np.zeros_like(filterTrace[k,:])

                    if  Moyenne_Couleur[2] > 150 :
                        if Moyenne_Couleur[0] > Moyenne_Couleur[2] :
                            filterTrace[k,:] = filterTrace[k,:] 
                        else :
                            filterTrace[k,:] = np.zeros_like(filterTrace[k,:])
                    """
                    
                    if Moyenne_Couleur[0] > Moyenne_Couleur[2] :
                        filterTrace[k,:] = filterTrace[k,:] 
                    else :
                        filterTrace[k,:] = np.zeros_like(filterTrace[k,:])
                
                    """
                    if Moyenne_Couleur[2] < 180 :                    
                        filterTrace[k,:] = filterTrace[k,:] 
                    else :
                        filterTrace[k,:] = np.zeros_like(filterTrace[k,:])
                    
                    
                    # CODE TEST : Pour supprimer les pixels avec des valeurs d'amplitude très excessive . 
                    fft_signal = np.fft.fft(filterTrace[k,:])
                    frequencies = np.fft.fftfreq(len(fft_signal), timeTrace[1] - timeTrace[0])
                    
                    power_spectrum = np.abs(fft_signal)**2 # Calculez le spectre de puissance
                    max_power = np.max(power_spectrum)  # Trouvez la valeur de la puissance maximale
                    
                    if max_power > 40000 : 
                        filterTrace[k,:] = np.zeros_like(filterTrace[k,:])  # Ou utilisez np.zeros_like(signal) pour mettre à 0
                    else:
                        filterTrace[k,:] = filterTrace[k,:] 
                    """
            # ------------ FILTRAGE  -----------------
            #for k in range(3): 
                #filterTrace[k,:] = filtfilt(b, a, filterTrace[k,:]) #Filtre Passe Bande ou Passe Bas   
                
            # ------------ CHANNEL COMBINATION ------------   
            #if np.all(filterTrace != 0):   
            if cf.METHOD == 'Green':              
                pulseTrace = filterTrace[1,:]
            elif cf.METHOD == 'G-R':
                pulseTrace = filterTrace[1,:] - filterTrace[0,:]
            elif cf.METHOD == 'Chrom':
                pulseTrace = np.zeros(traceSize)
                for t in range(len(filterTrace[1]) - winLength + 1):
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
                #Test : Normalisation sur la totalité du signal, et non morceau par morceaux
                for t in range(0, filterTrace.shape[1] - winLength + 1):     
                    C_window = filterTrace[:, t:t+winLength-1]
                    mean_C = np.mean(C_window, axis=1)              # Calcul de la moyenne de chaque ligne de la matrice C
                    diag_mean_C = np.diag(mean_C)                   # Création de la matrice diagonale avec les moyennes calculées
                    inv_diag_mean_C = np.linalg.inv(diag_mean_C)    # Calcul de l'inverse de la matrice diagonale
                    Cn = np.dot(inv_diag_mean_C, C_window)          # Calcul de Cn en appliquant la normalisation temporelle
                
                    S = np.array([[0, 1, -1],[-2, 1, 1]])
                    S = np.dot(S,Cn)
                    P = np.dot([1, np.std(S[0,:]) / np.std(S[1,:])], S)
                    #pulseTrace[t:t+winLength-1] += (P - np.mean(P))/np.std(P)  # overlap add
                    pulseTrace[t:t+winLength-1] += P


            elif cf.METHOD == 'POS2': #METHODE D'UNE PUBLICATION  #Xs = g(t) - b(t) & Ys = -2r(t) + g(t) + b(t)          
                    Xs = filterTrace[1,:] - filterTrace[2,:]
                    Ys = -2*filterTrace[0,:] + filterTrace[1,:] + filterTrace[2,:]
                    pulseTrace = Xs + (1)* Ys
                    #pulseTrace = Xs + (np.std(Xs) / np.std(Ys))* Ys

            pulseTrace = filtfilt(b, a, pulseTrace) #Filtre Passe Bande ou Passe Bas    

            if cf.SHORTFRQ == 1:
                for k in range(3):
                    ref_freq, h_freq = trouver_pics(pulseTrace[k, :], timeTrace)                                               # Trouves les fréquences 1ère et 2ème harmoniques
                    if ref_freq is not None and h_freq is not None:                                                             # Créer les bandes de fréquences
                        bande1 = (ref_freq - 0.1, ref_freq + 0.1)                                                               # Créer les bandes de fréquences 1er harmonique
                        bande2 = (h_freq - 0.1, h_freq + 0.1)                                                                   # Créer les bandes de fréquences 2ème harmonique  
                        frequencies = [bande1, bande2]                                                                          # Filtrer le signal dans les bandes de fréquences spécifiées
                        pulseTrace[k, :] = bandpass_filter(pulseTrace[k, :], frequencies, 1 / (timeTrace[1] - timeTrace[0]))  # Filtre Passe Bande FFT   
            
            crtTraces[m, n,:, :] = crtTrace         
            pulseTraces[m, n, :] = pulseTrace
    
    return crtTraces, pulseTraces

def squeeze_and_filter(mat, b, a):
    """
    Extrait les ma composante vert pour le pixel de coordonnée m,n sur les N images et effectue un filtre.

    Args:
        mat (ndarray as float32) : Tous les pixels de coordonne (m,n) sur les N images
        a (ndarray) : Coefficient Butterworth
        b (ndarray) : Coefficient Butterworth
    Return:
        crtTraces (ndArray)
        pulseTraces (ndArray)
    """
    crtTrace = np.squeeze(mat)              # extrait les compasantes RGB du pixel de coordonnée (m,n) sur les N images. Shape (3,N) 
    filterTrace = np.copy(crtTrace)         # Deep Copy
    pulseTrace = filterTrace[1,:]           # extrait la composante verte (G de RGB) du piwel (m,n) sur les N images. Shape (1,N)
    pulseTrace = filtfilt(b, a, pulseTrace) # Filtre Passe Bande ou Passe Bas   
    return crtTrace, pulseTrace

def sequential_refined(ROI, a, b):
    """
    Calcul original épuré crtTraces et pulseTraces 

    Args:
        ROI (ndarray as float32) : Region Of Interest
        a (ndarray) : Coefficient Butterworth
        b (ndarray) : Coefficient Butterworth
    Return:
        crtTraces (ndArray)
        pulseTraces (ndArray)
    """
    crtTraces = np.zeros((ROI.shape[0], ROI.shape[1],3, ROI.shape[3])).astype(np.float32)  
    pulseTraces = np.zeros((ROI.shape[0], ROI.shape[1], ROI.shape[3])).astype(np.float32) 
    for m in tqdm(range(ROI.shape[0]), desc="GetrPPG Original"): # Traiter chaque tranche de `ROI`
        for n in range(ROI.shape[1]):
            crtTrace, pulseTrace = squeeze_and_filter(ROI[m,n, 0:3, :], b, a)
            crtTraces[m, n,:, :] = crtTrace         # construction de crtTraces.   Shape(512, 512, 3, 300)
            pulseTraces[m, n, :] = pulseTrace       # construction de pulsetraces. Shape(512, 512, 300)

    return crtTraces, pulseTraces

def sequential_process_opti(ROI, a,b):
    """
    Calcul Multithread Opti de crtTraces et pulseTraces 

    Args:
        ROI (ndarray as float32) : Region Of Interest
        a (ndarray) : Coefficient Butterworth
        b (ndarray) : Coefficient Butterworth
    Return:
        crtTraces (ndArray)
        pulseTraces (ndArray)
    """
    results = Parallel(n_jobs=os.cpu_count())(                               # Declare autant the Job que de processeur
        delayed(squeeze_and_filter)(ROI[i, j, 0:3,:], b, a)                  
        for i in tqdm(range(ROI.shape[0]), desc="GetrPPG Opti")
        for j in range(ROI.shape[1])
    )
    crtTraces , pulseTraces = zip(*results)
    crtTraces   = np.array(crtTraces).reshape((ROI.shape[0], ROI.shape[1],3, ROI.shape[3]))
    pulseTraces = np.array(pulseTraces).reshape((ROI.shape[0], ROI.shape[1], ROI.shape[3]))

    return crtTraces, pulseTraces

def test_get_rppg(orignal, opti):
    """
    Test l'égalité entre la originale et la version optimisé du calcul de crtTraces et pulseTraces 
    Args : 
        original (tuple) : Tuple contenant crtTraces et pulseTraces avec le calcul non opti
        opti (tuple)     : Tuple contenant crtTraces et pulseTraces avec le calcul OPTI

    Raise : AssertionError
    """
    crtTraces , pulseTraces = zip(*orignal)
    crtTraces2 , pulseTraces2 = zip(*opti)

    for i in tqdm(range(pulseTraces.shape[0]), desc="Testing PulsesTraces equality"):
        for j in range(pulseTraces.shape[1]):
            for k in range(pulseTraces.shape[2]):
                    # Les 2 calculs ne se font pas à la meme précision de floatant d'ou le <1e-5. On regarde qu'ils sont à peu pres egaux
                    assert(abs(pulseTraces[i,j,k]-pulseTraces2[i,j,k]) <1e-5), f"Original and opti different at : {i},{j},{k} : {pulseTraces[i,j,k]}, {pulseTraces2[i,j,k]}"

    for i in tqdm(range(crtTraces.shape[0]), desc="Testing crtTraces esquality"):
        for j in range(crtTraces.shape[1]):
            for k in range(crtTraces.shape[2]):
                for l in range(crtTraces.shape[3]):
                    assert(crtTraces[i,j,k,l]==crtTraces2[i,j,k,l]), f"Original and opti different at : {i},{j},{k},{l}"


class RppgDescriptor():
    def __init__(self, res_path, fs, lowF, upF, filterOrder):
        self.res_path     = res_path
        self.fs           = fs
        self.lowF         = lowF
        self.upF          = upF
        self.filter_order = filterOrder

         
        
