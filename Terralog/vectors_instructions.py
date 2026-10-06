import json
import numpy as np
import ollama

# Le dictionnaire d'ancrage évolutif
ANCRES_ROUTAGE = {
    "culture": [
        "Je sème du blé dans le grand champ",
        "J'ai pulvérisé de l'azote à 100 litres par hectare",
        "Labour de la parcelle ce matin",
        "J'ai déchaumé cet après-midi",
        "Récolte du maïs",
        "Plantage de pomme de terre au niveau de la carrière"
    ],
    "elevage_intervention": [
        "J'ai fait un vaccin à la 4442",
        "Pesée du lot de génisses à 400 kilos",
        "Traitement antimammite administré sur la 5890"
    ],
    "elevage_naissance": [
        "Un petit veau est né de Marguerite",
        "Naissance d'une femelle cette nuit",
        "Achat d'un gros taureau"
    ],
    "elevage_mouvement_lot": [
        "J'ai changé les vaches de pré",
        "Déplacement du lot 3 vers le bâtiment B",
        "On a rentré les génisses pour l'hiver",
        "Transfert des veaux dans la nurserie"
    ],
    "materiel": [
        "J'ai vidangé le tracteur John Deere",
        "Le tank à lait est en panne"
    ],
    "elevage_sortie": [
        "La 4442 est partie à l'abattoir",
        "Vente du lot de veaux ce matin",
        "Une génisse est morte dans la nuit",
        "Départ de la vache 1298 pour la réforme",
        "Le vétérinaire a euthanasié le veau"
    ],
}

def generer_fichier_json():
    print("Génération de l'espace vectoriel de routage via Nomic...")
    vecteurs_export = []
    
    for categorie, phrases in ANCRES_ROUTAGE.items():
        for phrase in phrases:
            rep = ollama.embeddings(model='nomic-embed-text', prompt=phrase)
            vecteur = np.array(rep['embedding'])
            
            # Normalisation mathématique pour simplifier le calcul cosinus plus tard
            vecteur_norm = vecteur / np.linalg.norm(vecteur)
            
            vecteurs_export.append({
                "categorie": categorie,
                "phrase_reference": phrase, # Utile pour relire le JSON à l'œil nu
                "vecteur": vecteur_norm.tolist() # Indispensable : Numpy n'est pas nativement sérialisable en JSON
            })
            
    with open("vecteurs_routage.json", "w", encoding="utf-8") as f:
        json.dump(vecteurs_export, f, ensure_ascii=False, indent=2)
        
    print(f"Succès : {len(vecteurs_export)} vecteurs sauvegardés dans 'vecteurs_routage.json'.")

if __name__ == "__main__":
    generer_fichier_json()