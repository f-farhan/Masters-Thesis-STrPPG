"""
Description : Ce code permet de lire une séquence d'images, de les redimensionner si nécessaire, et d'appliquer un traitement d'image, en l'occurrence un filtre bilatéral.
Le résultat est enregistré sous forme d'un tenseur représentant la ROI (Région d'Intérêt).

Utilisation du GPU pour les filtres spaciaux (Plus rapide)

Le même code est utilisé pour le ROI principal et le ROI de référence.

 
Conversion de BGR à RGB effectuée ici.
Lecture des images au format 16 bits si spécifié dans le MAIN.
Définition manuelle des coefficients du filtre dans le MAIN.
Plusieurs méthodes de filtrage peuvent être implémentées, mais le filtre bilatéral reste l'un des meilleurs.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from config import cf
def getROI_Image_GPU(res_path, path_sources_img, d, sigmaColor, sigmaSpace,mode, rect, image_files, total_frames):
    
# =============== Création des Variables ===============
    n = 0
    object_frame = None
    fixed_width = rect[2]
    fixed_height = rect[3]
    
    resize_height = int(fixed_height//cf.RESIZE)
    resize_width = int(fixed_width//cf.RESIZE)

# =============== Test Opencv avec CUDA ===============
    if not cv2.cuda.getCudaEnabledDeviceCount():
        raise Exception("CUDA n'est pas supporté ou OpenCV n'a pas été compilé avec le support CUDA")
    
# =============== Traitement sur les images ===============
    ROI = process_images_gpu(res_path, path_sources_img, rect, image_files, total_frames, d, sigmaColor, sigmaSpace,mode,n,object_frame,resize_width,resize_height,RESIZE)

    return ROI

def process_images_gpu(res_path, path_sources_img, rect, image_files, total_frames, d, sigmaColor, sigmaSpace,mode,n,object_frame,resize_width,resize_height,RESIZE):

# =============== Création dossiers =============== 
    output_folder = os.path.join(res_path, f"output{mode}")
    os.makedirs(output_folder, exist_ok=True) 
    
# =============== Création de la matrice ROI pour chaque mode =============== 
    print(f'Création de la Matrice ROI {mode}')
    ROI = None

    for i, file in enumerate(tqdm(image_files[:total_frames], desc="Traitement des images sur GPU")):
    # ------------ Lecture des images ------------     
        if cf.FORMAT == 0:
            frame = cv2.imread(os.path.join(path_sources_img, file))                       # Lecture image en Uint8
        elif cf.FORMAT == 1:
            frame = cv2.imread(os.path.join(path_sources_img, file),cv2.IMREAD_UNCHANGED)  # Lecture image en Uint16   

        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                              # Conversion BGR to RGB
        frame = frame.astype(np.float32)                                            # Passage en float32 (= Nombre à virgule)

    # ------------ Redimensionnement ------------
        object_frame = frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]                                                  # Crop selon la selection 
        #resized_object_frame = object_frame[0:fixed_height:resize, 0:fixed_width:resize]                                       # Méthode 1 resize (Une ligne/Colone sur deux)
        resized_object_frame = cv2.resize(object_frame, (int(resize_width), int(resize_height)),interpolation=cv2.INTER_CUBIC)  # Méthode 2 Resize Opencv
        #resized_object_frame = cv2.pyrDown(object_frame)                                                                       # Méthode 3 PyrDown 
        resize_fixed_height, resize_fixed_width, _ = resized_object_frame.shape                                                 # Nouvelle shape de l'image 

    # ------------ Création du tenseur de la taille de la nouvelle shape ------------          
        if ROI is None:
            ROI = np.zeros((resize_fixed_height, resize_fixed_width, 3, total_frames), dtype=np.float32)                        

    # ------------ Traitements d'images ------------
        
        resized_object_frame_gpu = cv2.cuda_GpuMat()           # Crée un objet pour stocker des images sur le GPU
        resized_object_frame_gpu.upload(resized_object_frame)  # Télécharge l'image du CPU vers le GPU

        if cf.FORMAT == 1:
            sigmaColor16 = sigmaColor * (65535.0 / 255.0)    
            resized_object_frame_gpu = cv2.cuda.bilateralFilter(resized_object_frame_gpu, d, sigmaColor16, sigmaSpace)    # Filtre Bilateral UINT16
        else:
            resized_object_frame_gpu = cv2.cuda.bilateralFilter(resized_object_frame_gpu, d, sigmaColor, sigmaSpace) 
             
        resized_object_frame_gpu = resized_object_frame_gpu.download()                                              # Récupération du résultat depuis le GPU
        
        resized_object_frame_gpu = cv2.medianBlur(resized_object_frame_gpu, 3)                                              # Filtre flou Median  
                                                                
        #filtered_frame = cv2.medianBlur(filtered_frame, 3)
        
        ROI[:,:,:,n] = resized_object_frame_gpu                                                               # Ajout de la frame n dans le tenseur
        n += 1

        if cf.DEBUG == 1:
            resized_object_frame_gpu = cv2.cvtColor(resized_object_frame_gpu, cv2.COLOR_RGB2BGR)  
            if cf.FORMAT == 1:
                cv2.imwrite(os.path.join(output_folder, f"frame{i:04d}.png"), resized_object_frame_gpu/255)   # Enregistrement de l'image traitée UINT16
            else:
                cv2.imwrite(os.path.join(output_folder, f"frame{i:04d}.png"), resized_object_frame_gpu)       # Enregistrement de l'image traitée UNIT8

    if cf.DEBUG == 1:
        np.save(os.path.join(res_path, "npy_files", f"ROI{mode}.npy"), ROI)               # Enregistrement du tenseur ROI 
        print("Traitement terminé. Les images filtrées sont enregistrées dans :", output_folder)

    return ROI
