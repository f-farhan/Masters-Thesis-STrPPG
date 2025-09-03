"""
Description: MAIN ALGORITHM CODE
Allows launching the auxiliary .py files. 
 
To be specified: Name of the SOURCE folder, name of the Image folder. 
To be specified: Processing parameters, for example, whether the format is UINT8 or UINT16, the RESIZE size, the method used (Green, POS, Chrom, etc.).
To be specified: Signal filtering parameters = minimum frequency, maximum frequency, filter order. 
To be specified: Bilateral spatial filter parameters must be specified = d, sigmaColor, sigmaSpace.

"""

#=============== Import Libraries ===============#

import os
import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import tempfile

from pathlib import Path
from scipy.interpolate import interp1d
from multiprocessing import freeze_support
from UI import UI
from utils import write_to_notebook,sort
from OpenCSV import OpenCSV
from getROI_Image import getROI_Image
from getROI_Image_GPU import getROI_Image_GPU
from getrPPG import getrPPG
from getrPPGmean import getrPPGmean
from getHR import GtTrace_BPM_FFT
from getHRmean import getHRmean
from getPLOT import getPLOT
from getHR_MAE import mae_heatmap
from getSNR import snr_heatmap
from getAmplitudeFINDPEAKS import amplitude_heatmap_FINDPEAKS
from getAmplitudeFRQ import amplitude_heatmap_FRQ
from getAmplitudeLOCKIN import amplitude_heatmap_LOCKIN
from getAmplitudeNeurokit import amplitude_heatmap_neurokit
from getAmplitudeMEAN import amplitude_heatmap_mean
from getCorrelation import correlation_heatmap
from getPhase import phase_heatmap_hilbert
from getPhaseLOCKIN import phase_heatmap_LOCKIN
from getPowerMAP import power_map
from video_stabilization import auto_stabilize_return_video
from motion_artifact_reduction import normalize_rppg_tensor
from getSQI_template import compute_template_correlation_sqi
from getframe import save_first_frame
from heatmap_overlay import overlay_heatmap_with_manual_roi
from getHR import compute_power_spectrum
from getSource import *
from config import cf

#=============== MAIN ===============#

if __name__ == '__main__':
    print()        
    freeze_support()                #Required for multiprocessing 
    
# =============== Set Directories ===============
    Time_START = time.time()    #Time t0 - Start
        
    
    path_out = "C:/ff/outputs/CHROM/1"  # Set the output directory folder
    original_video_path = "E:/Faisal Farhan_Thesis/THESIS/Final Aquisition/RGB/Subject 1/temp_trimmed.avi" # Set the input path

    heatmap_path = 'C:/ff/outputs/CHROM/1/heatmaps/SNR_MAP_NO_BAR_0.png' # Set the output directory where the SNR MAP is located (This should be in the same folder of path_out/heatmaps)
    output_dir="C:/ff/outputs/CHROM/1/heatmaps" # Set the output directory of heatmaps 
    
    
 #============STABILIZATION=============#
       
    
    video_binary = auto_stabilize_return_video(original_video_path, output_dir=path_out)        

    with tempfile.NamedTemporaryFile(delete=False, suffix='.avi') as temp_stab_vid:        
        temp_stab_vid.write(video_binary)                                                   
        stabilized_video_path = temp_stab_vid.name                                          
    # Load source
    source = SourceFactory.create_source(stabilized_video_path, SourceType.VIDEO)           
    save_first_frame(original_video_path, path_out)                                       # Saves the first frame
    os.remove(stabilized_video_path)                                                        
 

#============Set Parameters=============# 

   #Set the frame rate
    fs=50       

    # Bilateral filter parameters
    d = 55        #35 | 12                             
    sigmaColor = 75 #60
    sigmaSpace = 75 #35

    # Butterworth filter parameters
    filtOrder = 2   
    lowF = 0.7     
    upF = 4

    ROI = None
    ROI_REF = None
    pulseTraces = None
    refTraces = None
    timeTrace = None 
       
#============Ground truth file=============#    
    filename = "gtdump.xmp"  
    gtHR = []; gtTrace = []; gtTime = []
    full_path = os.path.join(os.path.dirname(original_video_path), filename)

    try:
        data = OpenCSV(full_path, filename)
        gtTrace = data[0]; gtTime = data[2]; gtHR = data[1]
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{filename}' n'a pas été trouvé dans '{original_video_path}'.")
        gtTrace = np.zeros_like(gtTrace)
        gtTime = np.zeros_like(gtTime)
        gtHR = 0

#============Create directories and configuration file=============# 

    os.makedirs(os.path.join(path_out, "npy_files"), exist_ok=True) 
    os.makedirs(os.path.join(path_out, "heatmaps"), exist_ok=True)  
    path_to_notebook = Path(path_out) / "Configuration.txt"     # Configuration file   
    path_to_notebook.touch()                                    # Ensures the file exists, or creates the text file
      
    
# =============== ROIs selection ===============
    ROI_file = os.path.join(path_out, "npy_files/ROI.npy") 
    if not os.path.isfile(ROI_file): 
        print("ROI selection")     
        ROIs = {}
        Time_BeforeROI= time.time()

        if cf.SELECT_ROI == 1:
            cv2.namedWindow("ROI Selection", cv2.WINDOW_NORMAL)
            rect = cv2.selectROI("ROI Selection", source.get_frame(0))
        else:
            rect = (0, 0, source.width, source.height)
          
        ROI = getROI_Image(source,path_out, d, sigmaColor, sigmaSpace, rect)
        print("ROI selection for reference signal estimation")     
        img0 = ROI[:,:,:,0]

        if cf.SELECT_ROI_REF == 1:
            cv2.namedWindow("ROI selection for reference signal", cv2.WINDOW_NORMAL)
            rect_ref_image = cv2.selectROI("ROI selection for reference signal", cv2.cvtColor(img0.astype(np.uint8), cv2.COLOR_RGB2BGR))  
            cv2.destroyAllWindows() 
        else:  
            rect_ref_image = (0, 0, source.width, source.height)
                   
        write_to_notebook(path_out,"COEF FILTRE BILATERAL", f"Taille={d}, sigmaColor={sigmaColor}, sigmaSpace={sigmaSpace}") # Ecriture du COEF
        write_to_notebook(path_out,"FORMAT", "uint8" if cf.FORMAT == 0 else "uint16")
        
        x, y, w, h = rect_ref_image
        ROI_REF = ROI[y:y+h, x:x+w, :, :]        
        if cf.DEBUG == 1:
            np.save(os.path.join(path_out, "npy_files", f"ROI_REF.npy"), ROI_REF)  
        Time_AfterROI= time.time() 
    else:
         ROI = np.load(os.path.join(path_out,"npy_files/ROI.npy")) 
         ROI_REF = np.load(os.path.join(path_out,"npy_files/ROI_REF.npy")) 
        

# =============== Reference rPPG estimation ===============
    rppg_ref_file = os.path.join(path_out, "npy_files/rppg_ref.npy")      # refTraces file exist?
    if not os.path.isfile(rppg_ref_file):
        rppg_ref, timeTrace = getrPPGmean(ROI_REF,path_out,fs,lowF,upF,filtOrder)
        Time_AfterREF= time.time()
    else:
        rppg_ref = np.load(os.path.join(path_out,"npy_files/rppg_ref.npy")) 
        timeTrace = np.load(os.path.join(path_out,"npy_files/timeTrace.npy")) 


# =============== Plot FFT Spectrum of Reference Signal (once) ===============
    freqs, power_spectrum = compute_power_spectrum(rppg_ref, fs)
    dominant_index = np.argmax(power_spectrum)
    dominant_freq = freqs[dominant_index]
    dominant_power = power_spectrum[dominant_index]
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, power_spectrum, label='Power spectrum')
    plt.plot(dominant_freq, power_spectrum[dominant_index], 'ro', label=f'Dominant Frequency {dominant_freq:.5f} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('FFT Spectrum of Reference rPPG Signal')
    plt.legend()
    plt.grid(True)
    fft_save_path = os.path.join(path_out, "npy_files", "FFT_ref_signal.png")
    plt.savefig(fft_save_path, dpi=300)
    #plt.close()
    print(f"📈 FFT plot of reference signal saved to: {fft_save_path}")
         
# =============== Mean HR from rppg reference signal ===============
    mean_hr = getHRmean(rppg_ref,fs)
    print(f"Mean HR from reference signal: {np.round(mean_hr, 2)} bpm")
    Time_AfterHR= time.time()

# =============== rPPG signals estimation for each pixel of the ROI ===============
    pulseTraces_file = os.path.join(path_out, "npy_files/pulseTraces.npy")    #Fichier pulseTraces existe ?
    if not os.path.isfile(pulseTraces_file):
        pulseTraces = getrPPG(ROI,path_out,fs,lowF,upF,filtOrder)
        raw_pulseTraces = pulseTraces.copy()
        pulseTraces = normalize_rppg_tensor(pulseTraces, fs, cutoff=0.5)
        write_to_notebook(path_out,"METHOD", cf.METHOD)                                                        # Ecriture du type de METHODE       
        write_to_notebook(path_out,"FILTRAGE", f"Passe Bande: LowF={lowF}, HighF={upF}, Ordre={filtOrder}") # Ecriture des coefs du filtre
        write_to_notebook(path_out,"DETREND", "Non" if cf.DETREND == 0 else "Oui")                             # Ecriture si Detrend oui ou non
        write_to_notebook(path_out,"NORMALISATION", "Non" if cf.NORMALISATION == 0 else "Oui")                 # Ecriture si Normalisation oui ou non
        write_to_notebook(path_out,"FILTRE BANDE PASSANTE ETROIT", "Non" if cf.SHORTFRQ == 0 else "Oui")       # Ecriture si Passe bande étroit oui ou non
        write_to_notebook(path_out,"SUPPRESSION BACKGROUND", "Non" if cf.REMOVE_BACKGROUND == 0 else "Oui")    # Ecriture si suppression du background oui ou non
        Time_AfterRPPG= time.time()      
    
    pulse_ref = ROI_REF[:, :, 1, :]  # Green channel only, shape: (H, W, T)
    y, x = raw_pulseTraces.shape[0] // 2, raw_pulseTraces.shape[1] // 2                             # Choose a pixel from the ROI – center by default
    # Extract signals
    raw_signal = raw_pulseTraces[y, x, :]
    normalized_signal = pulseTraces[y, x, :]
    sqi_value = compute_template_correlation_sqi(normalized_signal, fs)
    plt.figure(figsize=(12, 5))
    plt.plot(raw_signal, label='Raw Signal', alpha=0.6)
    plt.plot(normalized_signal, label='Normalized Signal', alpha=0.8)
    plt.title(f"Pixel rPPG Signal at (x={x}, y={y})")
    plt.xlabel("Frame Index")
    plt.ylabel("Signal Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(path_out, "npy_files", f"signal_comparison_xy_{x}_{y}.png")
    plt.savefig(plot_path)
    plt.show()

    print(f"📈 Signal comparison plot saved to: {plot_path}")       
       
# =============== Computation of Heatmaps ===============
    if cf.HEATMAP == 1: 
        mae_heatmap(path_out, pulseTraces, timeTrace, fs, mean_hr, 1)                                # HEATMAP MAE (Erreur du BPM)
        correlation_heatmap(path_out, pulseTraces, timeTrace, refTraces,1)                           # HEATMAP Corrélation
        phase_heatmap_hilbert(path_out, pulseTraces, timeTrace,1)                                    # HEATMAP PHASE Hilbert
        phase_heatmap_LOCKIN(path_out, pulseTraces, timeTrace,mean_hr,refTraces, 1)                  # HEATMAP PHASE LOCKIN
        amplitude_heatmap_neurokit(path_out, pulseTraces, timeTrace, mean_hr, 1)
        amplitude_heatmap_FRQ(path_out, pulseTraces, timeTrace,mean_hr, lowF, upF,refTraces, 1)      # HEATMAP Energie en Frequentiel
        amplitude_heatmap_FINDPEAKS(path_out, pulseTraces, timeTrace,refTraces, 1)                   # HEATMAP FINDPEAKS
        amplitude_heatmap_LOCKIN(path_out, pulseTraces, timeTrace,mean_hr,refTraces,1)               # HEATMAP LOCKIN AMP
        amplitude_heatmap_mean(path_out, pulseTraces, timeTrace, 1)                                  # HEATMAP MEAN AMP
        snr_heatmap(path_out, pulse_ref, timeTrace, rppg_ref, 1, roi_coords=(x, y, w, h))            # HEATMAP SNR
        
    END= time.time()
    
# =============== Temps pour chaque partie =============== 
    """
    print("Temps Total: " + str(END - Time_START) + 
      "\nTemps de calcul ROI: " + str(Time_AfterROI - Time_BeforeROI) + 
      "\nTemps de calcul REF: " + str(Time_AfterREF - Time_AfterROI) + 
      "\nTemps de calcul HR: " + str(Time_AfterHR - Time_AfterREF) + 
      "\nTemps de calcul rPPG: " + str(Time_AfterRPPG - Time_AfterHR) + 
      "\nTemps de calcul HEATMAP: " + str(END - Time_AfterRPPG))"""

# =============== Display Plot =============== 
    if cf.AFFICHAGE == 1:
    #------------ BPM CALCULE vs BPM GTDump Oxymetre ------------
        mean_bpm_gtTrace = GtTrace_BPM_FFT(gtTime,gtTrace,fs)
        mean_bpm_sensor = np.mean(gtHR)
        print("Valeur du BPM Moyen  gtTrace", mean_bpm_gtTrace)
        print("Valeur du BPM Moyen du sensor", mean_bpm_sensor)
        print("Valeur du BPM Moyen du signal de REFERENCE", mean_hr)

        erreur_absolue = abs(mean_bpm_sensor - mean_hr)
        pourcentage_erreur = (erreur_absolue / mean_bpm_sensor) * 100
        print("Pourcentage d'erreur :", pourcentage_erreur)
        print(f"📊 Template Correlation SQI: {sqi_value:.3f}")

# =============== Heatmap Overlay ===============
    snr_image_path = os.path.join(path_out, "heatmaps", "SNR_MAP_NO_BAR_0.png") # The file names should be here by default
    hand_image_path = os.path.join(path_out, "roi_ref_visual.png")              # The file names should be here by default
    save_path = os.path.join(path_out, "heatmaps", "Overlay_SNR_on_Hand.png")   # The file names should be here by default
if os.path.exists(snr_image_path) and os.path.exists(hand_image_path):
    overlay_heatmap_with_manual_roi(
        original_img_path=hand_image_path,
        snr_map_path=snr_image_path,
        alpha=0.3,
        save_path=save_path
    )

#=========END=======================
