"""
Description : Code Obselète ! Premier code pour la gestion des ROI avec une séquence d'image. Test des différentes méthodes de filtre spatiaux. 
Ce code permet de lire une séquence d'images, de les redimensionner si nécessaire, et d'appliquer un traitement d'image, en l'occurrence un filtre bilatéral.
Le résultat est enregistré sous forme d'un tenseur représentant la ROI (Région d'Intérêt).
Test des méthodes de tracking et suivi. 

Utilisation du CPU pour les filtres spaciaux (Moins rapide)

Le même code est utilisé pour le ROI principal et le ROI de référence.

 
Conversion de BGR à RGB effectuée ici.
Lecture des images au format 16 bits si spécifié dans le MAIN.
Définition manuelle des coefficients du filtre dans le MAIN.
Plusieurs méthodes de filtrage peuvent être implémentées, mais le filtre bilatéral reste l'un des meilleurs.
"""

import os
import cv2
import numpy as np
import bm3d
from tqdm import tqdm
from skimage import filters
from skimage.filters import rank
from skimage.morphology import disk

from utils import build_gaussian_pyramid,reconstruct_from_pyramid
from config import cf
from dataclasses import dataclass


@dataclass
class RoiDescriptor:
    d             : int
    sigmaColor    : int
    sigmaSpace    : int
    rect          : np.ndarray

@dataclass
class Roi:
    nb_frames : int
    width     : int
    height    : int
    nb_channel: int
    frames    : np.ndarray 
        
        

# ============ Fonction Principal ============ 
def getROI_Image(source,path_out, d, sigmaColor, sigmaSpace, rect):

# =============== Création des Variables ===============   
    n = 0
    object_frame = None
    fixed_width = rect[2]
    fixed_height = rect[3]

    resize_height = int(fixed_height//cf.RESIZE)
    resize_width = int(fixed_width//cf.RESIZE)
    
# =============== Création de la matrice ROI ===============  
    print('ROI Creation')
    ROI = None

    for frame in tqdm(source.frames, desc="Applying spatial filters to all frames"):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) 

    # ------------ Redimensionnement ------------
        x, y, w, h = rect
        object_frame = frame[y:y+h, x:x+w]                                                  # Crop selon la selection 
        resized_object_frame = cv2.resize(object_frame, (int(resize_width), int(resize_height)),interpolation=cv2.INTER_CUBIC)  
        resize_fixed_height, resize_fixed_width, _ = resized_object_frame.shape  

    # ------------ Création du tenseur de la taille de la nouvelle shape ------------       
        if ROI is None:
            ROI = np.zeros((resize_fixed_height, resize_fixed_width, 3, source.nb_frames), dtype=np.float32)                        

    # ------------ Traitements d'images ------------ 
    
        if cf.GPU == 0: # on CPU (default case)
            # Bimateram filtering
            if cf.FORMAT == 1:
                sigmaColor16 = sigmaColor * (65535.0 / 255.0)    
                resized_object_frame = cv2.bilateralFilter(resized_object_frame, d, sigmaColor16, sigmaSpace)    # Filtre Bilateral UINT16
            else:
                resized_object_frame = cv2.bilateralFilter(resized_object_frame, d, sigmaColor, sigmaSpace)                                            
            #filtered_frame = cv2.medianBlur(filtered_frame, 3)
       
        else: #  GPU acceleration  
            resized_object_frame_gpu = cv2.cuda_GpuMat()           
            resized_object_frame_gpu.upload(resized_object_frame) 

            if cf.FORMAT == 1:
                sigmaColor16 = sigmaColor * (65535.0 / 255.0)    
                resized_object_frame_gpu = cv2.cuda.bilateralFilter(resized_object_frame_gpu, d, sigmaColor16, sigmaSpace)    
            else:
                resized_object_frame_gpu = cv2.cuda.bilateralFilter(resized_object_frame_gpu, d, sigmaColor, sigmaSpace) 
                
            resized_object_frame = resized_object_frame_gpu.download()                                             
            #resized_object_frame_gpu = cv2.medianBlur(resized_object_frame_gpu, 3)  


        ROI[:,:,:,n] = resized_object_frame              
        n += 1       


    # ------------ Affichage du traitement image  ------------
        #cv2.imshow("Object Tracker", (resized_object_frame/65535.0))
        #cv2.imshow("Object Tracker",resized_object_frame/255)

    # ------------ Enregistrement des images dans le dossier  ------------
        # if DEBUG == 1:
        #     resized_object_frame = cv2.cvtColor(resized_object_frame, cv2.COLOR_RGB2BGR)  
        #     if FORMAT == 1:
        #         cv2.imwrite(os.path.join(output_folder, f"frame{i:04d}.png"), resized_object_frame/255)   # Enregistrement de l'image traitée UINT16
        #     else:
        #         cv2.imwrite(os.path.join(output_folder, f"frame{i:04d}.png"), resized_object_frame)       # Enregistrement de l'image traitée UNIT8
    
    if cf.DEBUG == 1:
        np.save(os.path.join(path_out, "npy_files", f"ROI.npy"), ROI)       
    return ROI

 
def get_roi_image(source,res_path, roi_descriptor):

# =============== Création des Variables ===============   
    n = 0
    object_frame = None
    rect = roi_descriptor.rect
    fixed_width = roi_descriptor.rect[2]
    fixed_height = roi_descriptor.rect[3]

    resize_height = int(fixed_height//cf.RESIZE)
    resize_width = int(fixed_width//cf.RESIZE)

    roi_init =np.zeros((resize_height, resize_width, 3, source.nb_frames), dtype=np.float32)   
    roi  = Roi(source.nb_frames, fixed_width, fixed_height, 3, roi_init)
# =============== Création dossiers =============== 
    output_folder = os.path.join(res_path, f"output")
    os.makedirs(output_folder, exist_ok=True)

# =============== Création de la matrice ROI ===============  
    for frame in tqdm(source.frames, desc="Traitement des images sur CPU"):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                              # Conversion BGR to RGB
        frame = frame.astype(np.float32) # Passage en float32 

    # ------------ Redimensionnement ------------
        object_frame = frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]                                                  # Crop selon la selection 
        #resized_object_frame = object_frame[0:fixed_height:resize, 0:fixed_width:resize]                                       # Méthode 1 resize (Une ligne/Colone sur deux)
        resized_object_frame = cv2.resize(object_frame, (int(resize_width), int(resize_height)),interpolation=cv2.INTER_CUBIC)  # Méthode 2 Resize Opencv
        #resized_object_frame = cv2.pyrDown(object_frame)                                                                       # Méthode 3 PyrDown 
        resize_fixed_height, resize_fixed_width, _ = resized_object_frame.shape                                                 # Nouvelle shape de l'image 

    # ------------ Traitements d'images ------------             
        resized_object_frame = cv2.bilateralFilter(resized_object_frame, roi_descriptor.d, roi_descriptor.sigmaColor, roi_descriptor.sigmaSpace)          
        resized_object_frame = cv2.medianBlur(resized_object_frame, 3)  

        roi.frames[:,:,:,n] = resized_object_frame
        n += 1       
        
    return roi#, output_folder


def test_roi_eq(roi_ori, roi_opti):
    assert(roi_ori.shape==roi_opti.frames.shape), f"Original and Opti ROI should have the same size expect {roi_ori.shape} got {roi_opti.frames.shape}"    
    print(f"roi shape {roi_ori.shape}")
    print("[OK] ROI have the same Shape")
    for i in range(roi_ori.shape[0]):
        for j in range(roi_ori.shape[1]):
            for k in range(roi_ori.shape[2]):
                for l in range(roi_ori.shape[3]):
                    assert(roi_ori[i,j,k,l]== roi_opti.frames[i,j,k,l]), f"Roi should be same at {i}, {j}, {k},{l} got {roi_opti.frames[i,j,k,l]} expect {roi_ori[i,j,k,l]}"
    print("[OK] ROIs are the same")
    
