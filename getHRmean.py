from getHR import BPM_FFT_simple

def getHRmean(refTraces,fs):
   
    mean_bpm = BPM_FFT_simple(refTraces,fs,0) 
    
    return mean_bpm


     
