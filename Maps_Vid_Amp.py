"""
Description : Permet de calculer les cartes d'amplitudes, une par frame, grace à une fenêtre glissante.
Pas de calcul avec la méthode LOCKIN, car nos signaux sont normalisé et nous n'avons pas de valeur négative avec cette méthode.
On fait simplement la moyennne des valeurs de la fenêtre en cours. On suit le signal. 
On peux générer les maps seulement si nous avons calculer le fichier pulseTraces.

 
Définition de la durée de la séquence d'image manuellement 
Définition de la duré de la fenêtre manuellement 
Définition de la vmax des cartes manuellement (Normalisation)
"""

import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm

path_sources = "../Resultats/RES_Data18-Normal-YB.3"    #<- À définir 

# Chargement des signaux
pulseTraces, timeTrace, = (
    np.load(os.path.join(path_sources, "npy_files/pulseTraces.npy")),
    np.load(os.path.join(path_sources, "npy_files/timeTrace.npy"))
)

start_time = 0      # Début de la période [s]
end_time = 10     # Fin de la période [s]             <- À définir 
num_points = 300  # Nombres d'images de la séquence   <- À définir 

# Paramètres de la fenêtre glissante                    <- À définir 
taille_fenetre = 0.034    # POUR 30 fps Calcul: 1/30 fps    
#taille_fenetre = 0.00527 # POUR 190fps 


idx_start = np.argmin(np.abs(timeTrace - start_time))   # Trouvez les indices les plus proches pour les temps de début 
idx_end = np.argmin(np.abs(timeTrace - end_time))       # Trouvez les indices les plus proches pour les temps de fin


step = max(1, (idx_end - idx_start) // (num_points-1))  # Calculez le pas d'indexation en veillant à éviter une étape de 0

segment_times = np.linspace(start_time, end_time, num_points)
debut_fenetre = segment_times[0] 
fin_fenetre = debut_fenetre + taille_fenetre



for frame in tqdm(range(num_points), desc='Création des heatmaps', unit='heatmap'): # Boucle à travers les fenêtres de temps
    
    debut_fenetre = segment_times[frame % num_points]                               # Le déplacement de la fenêtre est maintenant d'un point à chaque frame
    fin_fenetre = debut_fenetre + taille_fenetre
    
    mask = (timeTrace >= debut_fenetre) & (timeTrace <= fin_fenetre)                # Sélection des données dans la fenêtre glissante en cours
    m, n, t = pulseTraces.shape   
    amp_map = np.zeros((m, n))                                                      # On crée une carte de phase pour chaque frame
    
    
    for i in range(m):
        for j in range(n):
            windowed_signal = pulseTraces[i, j, mask] * -1  # *-1 Inversion du signal car les pics du signal ppg correspond à un moment ou il y a peu de sang 
            if windowed_signal.size > 0:                    # Vérifie si le signal dans la fenêtre n'est pas vide
                amp_map[i, j] = np.mean(windowed_signal)    # Calcul de la moyenne du signal dans la fenêtre
            else:                                           # Si pas de données dans la fenêtre
                amp_map[i, j] = np.nan                      # Suppression du fond     
             
    cmap = cm.get_cmap('jet')               # Colormap
    normalized_amp_map = amp_map / 0.2     # <- À définir (Normalisation Basique) 

    plt.figure()
    plt.imshow(normalized_amp_map, cmap=cmap, origin='upper', aspect='equal',vmin=0,vmax=1) # <- À définir vmax si pas de normalisation
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(path_sources, "heatmaps", f'heatmap_{frame}.png'), format='png', bbox_inches='tight', pad_inches=0,dpi=500)
    plt.close()