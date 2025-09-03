import os
from getROI_Image import get_roi_image, RoiDescriptor
from getrPPG import RppgDescriptor, get_rppg, getrPPG
from getrPPGmean import get_rppg_mean
from getHRmean import getHRmean
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

from getSource import *
from config import cf
from ComputeStep import Step

if __name__ == "__main__":
    # Source
    #res_path         = "./resultats"
    #os.makedirs(os.path.join(res_path, "fichier_npy"), exist_ok=True)
    #path_sources     = "./Data18-Normal-YB"
    #path_sources_img = os.path.join(path_sources, "2023_09_05-13_51_31")

    path_out = "/Users/yannick/Desktop/vid_mainMS/out"
    path_source = "/Users/yannick/Desktop/vid_mainMS/img_in_Yannick/channel_0"
    
    # Recuperation des sources
    step_source = Step("Source", SourceFactory.create_source, True)
    source = step_source.run(path_source, SourceType.IMAGE)
    
    # Recuperation de la ROI
    rect = (0, 0, source.width, source.height)
    rect = np.array(rect)

    roi_descriptor = RoiDescriptor(d             = 55,
                                   sigmaColor    = 75,
                                   sigmaSpace    = 75,
                                   rect          = rect)

    step_roi = Step("ROI", get_roi_image, False)
    roi = step_roi.run(source, path_out, roi_descriptor)

    # Moyennage du ROI     
    rppg_descriptor = RppgDescriptor(path_out,
                                     fs          = 30,
                                     lowF        = 0.7,
                                     upF         = 4,
                                     filterOrder = 2)
    step_rppg_mean      = Step("RPPG_MEAN", get_rppg_mean, False)
    refTraces, timeTrace = step_rppg_mean.run(source, rppg_descriptor, roi)  

    # del source
    # BPM Moyen RPPG
    step_mean_bpm  = Step("HR MEAN", getHRmean, False)
    mean_bpm       = step_mean_bpm.run(path_out, refTraces, rppg_descriptor.fs)
    
    # Calcul Rppg
    step_rppg   = Step("RPPG",getrPPG , True)
    pulseTraces = step_rppg.run(roi.frames,
                                res_path,
                                rppg_descriptor.fs,
                                rppg_descriptor.lowF,
                                rppg_descriptor.upF,
                                rppg_descriptor.filter_order)
    
    mae_heatmap(res_path, pulseTraces, timeTrace, rppg_descriptor.fs, mean_bpm, 1)                                # HEATMAP MAE (Erreur du BPM)
    correlation_heatmap(res_path, pulseTraces, timeTrace, refTraces,1)                            # HEATMAP Corrélation
    snr_heatmap(res_path, pulseTraces, timeTrace,refTraces, 1)                                   # HEATMAP SNR
    phase_heatmap_hilbert(res_path, pulseTraces, timeTrace,1, refTraces)                                    # HEATMAP PHASE Hilbert
    phase_heatmap_LOCKIN(res_path, pulseTraces, timeTrace,mean_bpm,refTraces, 1)                  # HEATMAP PHASE LOCKIN
    amplitude_heatmap_neurokit(res_path, pulseTraces, timeTrace, mean_bpm, 1, refTraces)
    amplitude_heatmap_FRQ(res_path, pulseTraces, timeTrace,mean_bpm, rppg_descriptor.lowF, rppg_descriptor.upF,refTraces, 1)      # HEATMAP Energie en Frequentiel
    amplitude_heatmap_FINDPEAKS(res_path, pulseTraces, timeTrace,refTraces, 1)                    # HEATMAP FINDPEAKS
    amplitude_heatmap_LOCKIN(res_path, pulseTraces, timeTrace,mean_bpm,refTraces,1)               # HEATMAP LOCKIN AMP
    amplitude_heatmap_mean(res_path, pulseTraces, timeTrace, 1) 
