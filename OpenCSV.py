"""
Description : Code permettant la lecture du fichier de vérité terrain avec l'oxymètre de pouls.
Deux méthodes sont possibles :

Le fichier type "gtdump.xmp" est le plus récent et est utilisé avec le logiciel d'acquisition actuel.
Le fichier type "ground_truth_subj.txt" est une version plus ancienne du fichier de vérité terrain.

 
Le choix du fichier est défini dans le fichier MAIN, via la détection du nom du fichier. 
"""

import re
import numpy as np

def OpenCSV(full_path,filename): 
        
    if filename == "gtdump.xmp":  
        gtdata = np.loadtxt(full_path, delimiter=',') # Lire les données CSV

        # Extraire les colonnes
        gtTrace = gtdata[:, 3]
        gtTime = gtdata[:, 0] / 1000
        gtHR = gtdata[:, 1]

        # Normaliser les données (moyenne nulle et variance unitaire)
        gtTrace = gtTrace - np.mean(gtTrace)    
        gtTrace = gtTrace / np.std(gtTrace)
        data = [gtTrace, gtHR, gtTime]  # Retourner les variables nécessaires

        return data
       

    elif filename == "ground_truth_subj.txt" :

        with open(full_path) as fichier:        # Ouverture du fichier
            lignes = fichier.readlines()        # Lecture de toutes les lignes du fichier
            donnees = []                        # Initialisation du tableau des données
            regex = r"[-+]?\d*\.\d+e[+-]\d+"    # Expression régulière pour détecter les nombres sous forme scientifique
            
            for ligne in lignes:                # Parcours de chaque ligne
                ligne = re.sub(r"\s*(-\s*)(?=\d)", r"\1", ligne)        # Correction de la syntaxe des nombres négatifs 
                nombres = re.findall(regex, ligne)                      # Recherche de tous les nombres sous forme scientifique dans la ligne
                donnees.append([float(nombre) for nombre in nombres])   # Conversion en nombre flottant et stockage dans le tableau des données

            for ligne in donnees:               # Parcours de chaque nombre dans le tableau des données
                for nombre in ligne:   
                    if "e" in str(nombre):      # Extraction de l'exposant si le nombre contient "e"
                        exposant = int(str(nombre).split("e")[1])
                    else:
                        exposant = 0
                    puissance = pow(10, exposant)   # Calcul de la puissance de dix
                    nombre = nombre / puissance     # Multiplication par la puissance de dix
        return donnees
