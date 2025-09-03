"""
Description : Fichier py pour l'interface utilisateur, avec le pop up de sélection du dossier de travail. 

 
"""

import os
import tkinter as tk
from tkinter import ttk

outFolder = '/Users/yannick/Desktop/data/'

def UI(folder_name,):
    root = tk.Tk()
    root.title("Dossiers de résultats")

    
    res_path = os.path.join(outFolder, "Resultats", folder_name)
    print(f"the path is {res_path}")
    # Interface utilisateur
    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    label = tk.Label(frame, text=f"Dossier : {folder_name}")
    label.pack(side=tk.TOP, pady=(0, 10))

    # Combobox pour les suffixes
    subfolders = [f.replace(folder_name, '') for f in os.listdir(os.path.join(outFolder, "Resultats")) if f.startswith(folder_name) and os.path.isdir(os.path.join(os.path.join(outFolder, "Resultats"), f))]
    combo = ttk.Combobox(frame, values=subfolders, state="readonly", width=15)
    combo.set(subfolders[0] if subfolders else '')
    combo.pack(side=tk.LEFT)

    selected_path = [None]  # Utiliser une liste pour capturer le chemin à l'intérieur des fonctions

    def create_new_folder():
        i = 1
        while os.path.exists(f"{res_path}.{i}"):
            i += 1
        new_res_path = f"{res_path}.{i}"
        os.mkdir(new_res_path)
        print(f"Nouveau dossier créé : {new_res_path}")
        combo['values'] = [f.replace(folder_name, '') for f in os.listdir(os.path.join(outFolder, "Resultats")) if f.startswith(folder_name) and os.path.isdir(os.path.join(os.path.join(outFolder, "Resultats"), f))]
        combo.set(f".{i}")

    def select_folder():
        selected_path[0] = os.path.join(os.path.join(outFolder, "Resultats"), f"{folder_name}{combo.get()}")
        root.destroy()

    create_button = tk.Button(frame, text="Créer nouveau", command=create_new_folder)
    create_button.pack(side=tk.RIGHT, padx=(10, 0))

    select_button = tk.Button(frame, text="Sélectionner", command=select_folder)
    select_button.pack(side=tk.RIGHT)

    root.mainloop()
    return selected_path[0]