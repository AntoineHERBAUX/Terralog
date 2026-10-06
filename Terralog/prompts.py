"""
prompts.py — Prompts et exemples few-shot de TerraLog.

Les exemples sont construits avec les classes Pydantic de schemas.py : si un schéma
change, un exemple obsolète lève une erreur à l'import au lieu de dériver en silence.

ATTENTION : ces phrases d'exemple ne doivent PAS se retrouver dans ton jeu de test,
sinon tu mesures la mémoire du modèle et non sa capacité à généraliser.
"""
import json
import sys
import os

# Ajouter le répertoire Terralog au path si ce n'est pas déjà fait
# Cela permet d'importer les modules locaux quel que soit le répertoire de travail actuel
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from schemas import (
    DESCRIPTION_CATEGORIES,
    Consultation,
    Culture,
    Intervention,
    Materiel,
    MouvementLot,
    Naissance,
    Routage,
    Sortie,
    champs_du_modele,
)


def _o(modele, **valeurs):
    """Construit un exemple : tout champ non cité vaut null (ou [] pour les listes)."""
    base = {nom: ([] if nom == "numeros_boucles" else None) for nom in modele.model_fields}
    inconnus = set(valeurs) - set(base)
    assert not inconnus, f"{modele.__name__} : champs inconnus {inconnus}"
    base.update(valeurs)
    return modele(**base)


# ──────────────────────────────────────────────────────────────────────────────
# Prompts système de l'expert : un socle commun + 2-3 lignes par catégorie
# ──────────────────────────────────────────────────────────────────────────────
PROMPT_BASE = (
    "Tu extrais des informations d'une phrase dictée par un agriculteur. "
    "La phrase vient d'une transcription automatique et peut contenir des erreurs. "
    "Règles : "
    "1) Remplis uniquement ce qui est dit. Ce qui n'est pas dit vaut null (liste vide pour une liste). N'invente jamais. "
    "2) Recopie les noms propres et les numéros tels qu'ils sont dits. "
    "3) Pour la date, recopie les mots dits dans expression_date. Ne calcule aucune date. "
    "4) Si on te donne des données actuelles et une question, l'agriculteur répond à cette question "
    "ou corrige une valeur : modifie uniquement les champs concernés et recopie les autres à l'identique."
)

SPECIFIQUE = {
    "culture": (
        "Contexte : travaux dans les champs. "
        "Un engrais (azote, ammonitrate...) = fertilisation. Un fongicide, herbicide ou insecticide = traitement_phyto. Labourer = labour. Déchaumer = déchaumage. Récolter = récolte"
    ),
    "elevage_intervention": (
        "Contexte : acte sur un animal identifié par son numéro de boucle. "
        "Un vaccin, un antibiotique ou un traitement = soin_sanitaire. Une pesée = suivi_croissance_pesee."
    ),
    "elevage_naissance": (
        "Contexte : naissance ou arrivée d'un animal. "
        "Le numéro de boucle est celui du nouveau-né, jamais celui de sa mère : "
        "un numéro donné comme mère va dans le champ mere."
    ),
    "elevage_sortie": (
        "Contexte : sortie d'un animal du troupeau. "
        "Choisis un motif seulement s'il est dit ou évident (partir à l'abattoir = abattage)."
    ),
    "elevage_mouvement_lot": (
        "Contexte : déplacement d'animaux. "
        "Extrais TOUS les numéros de boucle dictés (4 chiffres chacun). "
        "Ne résume jamais par 'les animaux' ou 'les vaches'."
    ),
    "materiel": "Contexte : panne, entretien ou carburant d'une machine ou d'un équipement de la ferme.",
}

PROMPT_CONSULTATION = (
    "Tu comprends une question posée par un agriculteur sur sa ferme "
    "(la phrase vient d'une transcription automatique et peut contenir des erreurs). "
    "Tu NE réponds PAS à la question : tu indiques seulement ce qu'il demande (sujet), "
    "de quoi il parle (cible_type) et la référence citée (numéro de boucle, nom de parcelle ou de machine), "
    "tels qu'ils sont dits. Ce qui n'est pas dit vaut null. N'invente jamais."
)


# ──────────────────────────────────────────────────────────────────────────────
# Exemples few-shot : (phrase dictée, sortie attendue)
# ──────────────────────────────────────────────────────────────────────────────
EXEMPLES = {
    "culture": [
        ("Je sème.", _o(Culture, action_realisee="semis")),
        (
            "Hier soir j'ai épandu 150 kilos d'azote par hectare chez Patrick.",
            _o(Culture, parcelle_concernee="chez Patrick", action_realisee="fertilisation",
               intrant_ou_culture="azote", dose_valeur=150, dose_unite="kg/ha", expression_date="hier soir"),
        ),
        (
            "Récolte du colza terminée aux Grandes Terres.",
            _o(Culture, parcelle_concernee="Grandes Terres", action_realisee="recolte", intrant_ou_culture="colza"),
        ),
    ],
    "elevage_intervention": [
        ("J'ai fait un vaccin.", _o(Intervention, type_intervention="soin_sanitaire", produit_ou_materiel="vaccin")),
        (
            "La 3321 pèse 580 kilos.",
            _o(Intervention, numero_boucle="3321", type_intervention="suivi_croissance_pesee",
               dosage_ou_mesure="580 kilos"),
        ),
        (
            "Le vétérinaire a vacciné la 6150 ce matin.",
            _o(Intervention, numero_boucle="6150", type_intervention="soin_sanitaire",
               produit_ou_materiel="vaccin", intervenant="vétérinaire", expression_date="ce matin"),
        ),
    ],
    "elevage_naissance": [
        ("Un petit mâle vient de naître.", _o(Naissance, sexe="male")),
        (
            "Le veau 4402 est né de Marguerite, on l'appelle Pirate.",
            _o(Naissance, numero_boucle="4402", prenom="Pirate", espece="bovin", mere="Marguerite"),
        ),
        (
            "Un porcelet femelle est né hier dans la porcherie.",
            _o(Naissance, espece="porcin", sexe="femelle", localisation="porcherie", expression_date="hier"),
        ),
    ],
    "elevage_sortie": [
        ("Elle est morte.", _o(Sortie)),
        (
            "La 3702 est partie à l'abattoir de Lens hier.",
            _o(Sortie, numero_boucle="3702", motif_sortie="abattage",
               destination="abattoir de Lens", expression_date="hier"),
        ),
        (
            "La 5518 a été vendue à Martin.",
            _o(Sortie, numero_boucle="5518", motif_sortie="vente_elevage", destination="Martin"),
        ),
    ],
    "elevage_mouvement_lot": [
        (
            "J'ai mis la 2001, la 2002 et la 2003 dans la stabulation.",
            _o(MouvementLot, numeros_boucles=["2001", "2002", "2003"], lieu_arrivee="stabulation"),
        ),
        (
            "Transfert du lot des génisses de la prairie nord vers l'étable.",
            _o(MouvementLot, nom_lot="lot des génisses", lieu_depart="prairie nord", lieu_arrivee="étable"),
        ),
    ],
    "materiel": [
        ("J'ai une panne !", _o(Materiel, type_operation="panne")),
        (
            "Le tank à lait fuit depuis ce matin.",
            _o(Materiel, equipement="tank à lait", type_operation="panne",
               description_action="fuit", expression_date="ce matin"),
        ),
        (
            "J'ai changé le filtre à huile du Massey Ferguson hier.",
            _o(Materiel, equipement="Massey Ferguson", type_operation="entretien_regulier",
               description_action="changement filtre à huile", expression_date="hier"),
        ),
    ],
    "consultation": [
        (
            "Quelles vaches ne faut-il pas traire ce soir ?",
            _o(Consultation, sujet="animaux_en_delai", cible_type="troupeau"),
        ),
        (
            "C'est quand le dernier traitement de la 3790 ?",
            _o(Consultation, sujet="dernier_evenement", cible_type="animal", reference="3790",
               type_evenement="traitement"),
        ),
        (
            "Qu'est-ce que j'ai fait sur les Grandes Terres ?",
            _o(Consultation, sujet="historique", cible_type="parcelle", reference="Grandes Terres"),
        ),
        ("Montre-moi la 6150.", _o(Consultation, sujet="fiche", cible_type="animal", reference="6150")),
        (
            "La 2208 est en délai d'attente ?",
            _o(Consultation, sujet="delai_attente", cible_type="animal", reference="2208"),
        ),
        (
            "Quand est-ce que le Massey Ferguson est tombé en panne ?",
            _o(Consultation, sujet="dernier_evenement", cible_type="equipement", reference="Massey Ferguson",
               type_evenement="panne"),
        ),
    ],
}

# Un exemple de relance par catégorie : (données actuelles, question posée, réponse, résultat)
EXEMPLES_SESSION = {
    "culture": (
        _o(Culture, action_realisee="traitement_phyto", intrant_ou_culture="fongicide"),
        "Sur quelle parcelle êtes-vous intervenu ?",
        "Les Peupliers.",
        _o(Culture, parcelle_concernee="Les Peupliers", action_realisee="traitement_phyto",
           intrant_ou_culture="fongicide"),
    ),
    "elevage_intervention": (
        _o(Intervention, type_intervention="soin_sanitaire", produit_ou_materiel="vaccin"),
        "Quel est le numéro de boucle de l'animal concerné ?",
        "La 2208.",
        _o(Intervention, numero_boucle="2208", type_intervention="soin_sanitaire", produit_ou_materiel="vaccin"),
    ),
    "elevage_naissance": (
        _o(Naissance, espece="bovin", numero_boucle="7010"),
        "S'agit-il d'un mâle ou d'une femelle ?",
        "Une femelle.",
        _o(Naissance, espece="bovin", numero_boucle="7010", sexe="femelle"),
    ),
    "elevage_sortie": (
        _o(Sortie, numero_boucle="9012"),
        "Quelle est la cause de la sortie (vente, abattage, mort naturelle...) ?",
        "Elle est morte de maladie.",
        _o(Sortie, numero_boucle="9012", motif_sortie="maladie"),
    ),
    "elevage_mouvement_lot": (
        _o(MouvementLot, lieu_arrivee="pré"),
        "Quels animaux avez-vous déplacé ? Donnez les numéros de boucle ou le nom du lot.",
        "La 3110 et la 3111.",
        _o(MouvementLot, numeros_boucles=["3110", "3111"], lieu_arrivee="pré"),
    ),
    "materiel": (
        _o(Materiel, type_operation="panne", description_action="bruit anormal"),
        "Sur quel équipement ou machine êtes-vous intervenu ?",
        "Le tracteur Claas.",
        _o(Materiel, equipement="tracteur Claas", type_operation="panne", description_action="bruit anormal"),
    ),
}


def tour_session(etat: dict, question: str, reponse: str) -> str:
    """Format du message utilisateur quand une question est en attente (relance ou correction)."""
    return (
        f"Données actuelles : {json.dumps(etat, ensure_ascii=False)}\n"
        f"Question posée : {question}\n"
        f"Réponse de l'agriculteur : {reponse}"
    )


def messages_expert(categorie: str, texte: str, etat: dict = None, question: str = None) -> list:
    """Messages pour l'expert : système + exemples en vrais tours de chat + tour réel.

    etat/question : à fournir quand une question est en attente (relance ou correction).
    """
    if categorie == "consultation":
        # Question en un seul tour : pas de brouillon, pas de relance, pas de date
        messages = [{"role": "system", "content": PROMPT_CONSULTATION}]
        for phrase, objet in EXEMPLES["consultation"]:
            messages.append({"role": "user", "content": phrase})
            messages.append({"role": "assistant", "content": objet.model_dump_json()})
        messages.append({"role": "user", "content": texte})
        return messages

    messages = [{"role": "system", "content": f"{PROMPT_BASE} {SPECIFIQUE[categorie]}"}]

    for phrase, objet in EXEMPLES[categorie]:
        messages.append({"role": "user", "content": phrase})
        messages.append({"role": "assistant", "content": objet.model_dump_json()})

    etat_ex, question_ex, reponse_ex, resultat_ex = EXEMPLES_SESSION[categorie]
    messages.append({"role": "user", "content": tour_session(etat_ex.model_dump(), question_ex, reponse_ex)})
    messages.append({"role": "assistant", "content": resultat_ex.model_dump_json()})

    if etat is None:
        messages.append({"role": "user", "content": texte})
    else:
        # Seulement les champs du schéma : pas de date_action, statut, etc.
        etat_filtre = {c: etat.get(c) for c in champs_du_modele(categorie)}
        messages.append({"role": "user", "content": tour_session(etat_filtre, question or "", texte)})
    return messages


# ──────────────────────────────────────────────────────────────────────────────
# Routeur : (phrase brute, phrase corrigée, catégorie)
# ──────────────────────────────────────────────────────────────────────────────
PROMPT_ROUTEUR = (
    "Tu classes une phrase dictée par un agriculteur (transcription automatique, avec des erreurs possibles). "
    "Recopie d'abord la phrase en corrigeant les erreurs phonétiques évidentes "
    "(ex: 'antivamite' -> 'anti-mammite') sans jamais toucher aux chiffres, puis choisis la catégorie. "
    "Si la phrase raconte ce que l'agriculteur a fait à un animal, c'est de l'élevage. "
    "Si c'est une question ou une demande d'information (quand, quel, combien, montre-moi...), "
    "c'est une consultation, même si elle parle d'un animal. "
    + DESCRIPTION_CATEGORIES
)

EXEMPLES_ROUTAGE = [
    ("J'ai soigné une mammite avec de l'antivamite.", "J'ai soigné une mammite avec de l'anti-mammite.", "elevage_intervention"),
    ("Elle est morte.", "Elle est morte.", "elevage_sortie"),
    ("J'ai amené les vaches au pré.", "J'ai amené les vaches au pré.", "elevage_mouvement_lot"),
    ("J'ai fait la vidange du tracteur.", "J'ai fait la vidange du tracteur.", "materiel"),
    ("J'ai désherbé la parcelle du bas.", "J'ai désherbé la parcelle du bas.", "culture"),
    ("Il va faire beau demain.", "Il va faire beau demain.", "inconnue"),
    # Paires minimales : même sujet, une affirmation (on enregistre) et une question (on consulte)
    ("La 2615 a vêlé.", "La 2615 a vêlé.", "elevage_naissance"),
    ("Quand a vêlé la 2615 ?", "Quand a vêlé la 2615 ?", "consultation"),
    ("J'ai traité la 3790.", "J'ai traité la 3790.", "elevage_intervention"),
    ("Quel est le dernier traitement de la 3790 ?", "Quel est le dernier traitement de la 3790 ?", "consultation"),
    ("J'ai semé du blé sur le Grand Champ.", "J'ai semé du blé sur le Grand Champ.", "culture"),
    ("Montre-moi le Grand Champ.", "Montre-moi le Grand Champ.", "consultation"),
    ("Quelles vaches ne faut-il pas traire ce soir ?", "Quelles vaches ne faut-il pas traire ce soir ?", "consultation"),
]


def messages_routeur(texte: str) -> list:
    messages = [{"role": "system", "content": PROMPT_ROUTEUR}]
    for phrase, corrigee, categorie in EXEMPLES_ROUTAGE:
        messages.append({"role": "user", "content": phrase})
        messages.append({
            "role": "assistant",
            "content": Routage(phrase_corrigee=corrigee, categorie=categorie).model_dump_json(),
        })
    messages.append({"role": "user", "content": texte})
    return messages
