"""
Description : Code utilisé pour calculer en fréquentiel le BPM, incluant le BPM moyen et le BPM par pixel pour la carte de MAE, en appelant la fonction BPM_FFT_simple. 
Cette fonction est plus simple que celle utilisée dans l'archive ARCHIVE_getHR, où l'on faisait des petites fenêtres pour observer les variations du BPM dans le temps. 
La fonction a été simplifiée pour prendre en compte la totalité du signal. Nous n'obtenons plus les variations au cours du temps, mais le résultat est plus précis pour la moyenne.

Car finalement dans l'algorithme, il n'était pas utile d'avoir les variations du BPM au cours du temps.
La fonction avec find_peaks pour compter le nombre de pics pour une certaines durée n'a pas été rajouté car moins précis et pas utilisé.

Une fonction pour les signaux gtTrace de l'oxymètre de pouls a été ajoutée. Les signaux gtTraces n'ayant pas de fréquence fixe, il est difficile de réaliser un spectre en fréquence. 
Cependant, nous pouvons faire une interpolation avec la fréquence d'échantillonnage désirée pour obtenir le BPM moyen de la trace de l'oxymètre de pouls.

Le paramètre PLOT dans la fonction permet d'afficher le spectre du signal traité
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def BPM_FFT_simple(ppg_data, fs,PLOT):
    
    freqs, power_spectrum = compute_power_spectrum(ppg_data, fs)
    
    # Trouver la fréquence dominante dans tout le spectre
    dominant_index = np.argmax(power_spectrum)
    dominant_freq = freqs[dominant_index]
    dominant_power = power_spectrum[dominant_index]
    bpm = dominant_freq * 60 #if dominant_power > 500 else 0

    if PLOT == 1: 
        # Créer un plot pour afficher le spectre de puissance
        plt.figure(figsize=(10, 6))
        plt.plot(freqs, power_spectrum, label='Spectre de puissance')
        plt.plot(dominant_freq, power_spectrum[dominant_index], 'ro', label=f'Pic dominant à {dominant_freq:.5f} Hz')
        plt.xlabel('Fréquence (Hz)')
        plt.ylabel('Puissance')
        plt.title('Spectre de puissance de la FFT et fréquence dominante (Pixel selectionné)')
        plt.legend()
        plt.grid(True)
    
    return bpm
  
def compute_power_spectrum(x, Fs):
    N = len(x) * 3                          # Détermine la longueur de la FFT (Fast Fourier Transform) à utiliser
    freq = np.arange(N) * Fs / N            # Calcule les fréquences correspondantes à chaque échantillon de la FFT
    power = np.abs(np.fft.fft(x, N)) ** 2   # Calcule le spectre de puissance en effectuant la FFT sur le signal
    return freq[:N // 2], power[:N // 2]


def GtTrace_BPM_FFT(time_data, ppg_data, new_sampling_rate):

    if len(time_data) == 0 or len(ppg_data) == 0:
        # Il est possible de retourner une valeur spécifique ou de lever une exception personnalisée
        print("Erreur: Les données de temps ou les données PPG sont vides.")
        return 0  # Ou lever une exception, par exemple: raise ValueError("Les données sont vides")
    
    # Calculer la durée totale et définir la nouvelle grille de temps basée sur les bornes réelles de time_data
    new_time = np.arange(time_data[0], time_data[-1], 1/new_sampling_rate)
    
    # Créer une fonction d'interpolation
    interpolation_function = interp1d(time_data, ppg_data, kind='cubic', fill_value="extrapolate")
    
    # Interpoler les données
    new_ppg_data = interpolation_function(new_time)
    
    n = len(new_ppg_data)
    freqs = np.fft.fftfreq(n, 1/new_sampling_rate)
    fft_magnitude = np.abs(np.fft.fft(new_ppg_data))
    
    # Trouver les pics dans le spectre de magnitude de la FFT
    peaks, _ = find_peaks(fft_magnitude, height=np.max(fft_magnitude)/4)
    peak_freq = freqs[peaks][np.argmax(fft_magnitude[peaks])]
    bpm = 60 * peak_freq

    return bpm
