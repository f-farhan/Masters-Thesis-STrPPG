"""
Description : Code contenant plusieurs fonctions appelées dans le code principal et dans plusieurs autres modules. 
- Écriture dans le fichier txt
- Tri des images
- Detrend
- Filtre passe-bande FFT
- Filtre de lissage (moyenne mobile)
- Filtres spatiaux
- Détection des pics

 
"""
import os
import cv2 
import numpy as np
import scipy.sparse
from tqdm import tqdm
from scipy.signal import butter, filtfilt,find_peaks
from scipy.fft import fft, fftfreq


def write_to_notebook(res_path, name, value):
    file_path = os.path.join(res_path, "Configuration.txt") # Construire le chemin complet vers le fichier
    entry = f"{name}: {value}\n"                            # Préparer la ligne à ajouter

    if os.path.exists(file_path):                           # Vérifier si le fichier existe pour éviter une erreur lors de l'ouverture en mode lecture
        with open(file_path, "r") as notebook:              # Ouvrir le fichier en mode lecture pour vérifier le contenu existant
            if entry in notebook.readlines():
                return                                      # Ne rien faire si l'entrée existe déjà   
    with open(file_path, "a") as notebook:                  # Si l'entrée n'existe pas, ajouter au fichier
        notebook.write(entry)

# =============== Tri/ Rangement Images  ===============
def sort (file):
        base = os.path.basename(file)
        number = base.replace('img', '').split('.')[0]
        return int(number) 

# =============== Détrend (Neurokit) ===============
def detrend(signal, regularization=500):
    """Method by Tarvainen et al., 2002.
    - Tarvainen, M. P., Ranta-Aho, P. O., & Karjalainen, P. A. (2002). An advanced detrending method
    with application to HRV analysis. IEEE Transactions on Biomedical Engineering, 49(2), 172-175.
    """
    N = len(signal)                                                         # Récupère la longueur du signal
    identity = np.eye(N)                                                    # Crée une matrice identité de taille N
    #B = np.dot(np.ones((N - 2, 1)), np.array([[1, -2, 1]]))
    B = np.dot(np.ones((N, 1)), np.array([[1, -2, 1]]))                     # Crée une matrice B avec des valeurs 1, -2, 1 pour la dérivée seconde
    D_2 = scipy.sparse.dia_matrix((B.T, [0, 1, 2]), shape=(N - 2, N))       # Crée une matrice diagonale creuse représentant la dérivée seconde
    #D_2 = scipy.sparse.dia_matrix((B.T, [0, 1, 2]), shape=(N - 3, N-1))
    inv = np.linalg.inv(identity + regularization**2 * D_2.T @ D_2)         # Calcule l'inverse de la matrice identité + régularisation * dérivée seconde transposée * dérivée seconde
    z_stat = ((identity - inv)) @ signal                                    # Calcule la statistique z
    trend = np.squeeze(np.asarray(signal - z_stat))                         # Calcul de la tendance

    return signal - trend    

def detrend_pixels(ROI, regularization=500):
    height, width, _, total_frames = ROI.shape
    detrended_ROI = np.zeros_like(ROI)

    for m in tqdm(range(height), desc='Progression m', unit='row'):
        for n in tqdm(range(width), desc='Progression n', unit='column', leave=False):
            for c in range(3):
                signal = np.squeeze(ROI[m, n, c, :])
                detrended_signal = detrend(signal, regularization)
                detrended_ROI[m, n, c, :] = detrended_signal

    return detrended_ROI

# =============== Filtre Passe Bande FFT ===============
def bandpass_filter(signal, frequencies, fs):
    signal_fft = np.fft.fft(signal)                                                                         # Calcul de la transformée de Fourier du signal
    freqs = np.fft.fftfreq(len(signal), 1/fs)                                                               # Création du tableau des fréquences correspondant à la transformée de Fourier
    indices = np.logical_or.reduce([np.logical_and(freqs >= f[0], freqs <= f[1]) for f in frequencies])     # Filtrage passe-bande en fréquence
    signal_fft[np.logical_not(indices)] = 0                                                                 # Met à zéro les fréquences qui ne sont pas dans la Bande Passante 
    filtered_signal = np.fft.ifft(signal_fft)                                                               # Calcul de la transformée de Fourier inverse pour obtenir le signal filtré

    return (filtered_signal)

# =============== Trouves les fréquences 1ère et 2ème harmoniques ===============
def trouver_pics(signal, timeTrace):
    spectre = fft(signal)                                                           # Effectuer la transformée de Fourier
    freqs = fftfreq(len(signal), timeTrace[1] - timeTrace[0])                       # Calculer les fréquences correspondant aux composantes de la transformée de Fourier
    amplitudes = np.abs(spectre)                                                    # Calculer les amplitudes des composantes de la transformée de Fourier
    peaks, _ = find_peaks(amplitudes, height=np.max(amplitudes) * 0.1, distance=10) # Trouver les pics dans le spectre avec une certaine hauteur et une certaine distance entre eux
    sorted_peaks = sorted(peaks, key=lambda x: amplitudes[x], reverse=True)         # Trier les pics par amplitude décroissante

    if not sorted_peaks:                                                            # Vérifier si des pics ont été trouvés
        return None, None                                                           # ou gérer l'erreur d'une autre manière
    
    reference_freq = freqs[sorted_peaks[0]]                                         # Trouver la fréquence de référence (premier pic)

    harmonique2_freq = None
    for peak in sorted_peaks[1:]:                                                   # Trouver les fréquences du deuxième pics
        freq = freqs[peak]
        if freq > 0 and abs(freq - reference_freq) >= 0.7:                          # Vérifier l'éloignement par rapport à la première harmonique
            if harmonique2_freq is None:
                harmonique2_freq = freq
                break

    return reference_freq, harmonique2_freq


# =============== Filtre de lissage (moyenne mobile) ===============
def smooth_filter(signal, window_size):
    reflected_signal = np.pad(signal, (window_size//2, window_size//2), mode='reflect') # Padding du signal avec la méthode 'reflect' pour éviter les effets de bord  
    window = np.ones(window_size) / window_size                                         # Définition du noyau de lissage | moyenne mobile
    smoothed_signal = np.convolve(reflected_signal, window, mode='valid')               # Application du filtre de lissage avec np.convolve()
   
    return smoothed_signal

# =============== Filtre spatial ===============
def build_gaussian_pyramid(image,num_levels):
    pyramid = [image]
    for i in range(num_levels - 1):                 
        image = cv2.pyrDown(image) # Réduire l'image et l'ajouter à la pyramide
        pyramid.append(image)

    return pyramid

# =============== Filtre spatial ===============
def reconstruct_from_pyramid(pyramid):
    reconstructed_image = pyramid[-1]                       # Démarrer avec la plus petite image
    for i in range(len(pyramid) - 2, -1, -1):               # Commencer du dernier niveau
        size = (pyramid[i].shape[1], pyramid[i].shape[0])   # Taille de l'image actuelle
        reconstructed_image = cv2.pyrUp(reconstructed_image, dstsize=size)

    return reconstructed_image
