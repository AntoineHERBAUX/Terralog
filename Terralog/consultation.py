"""
consultation.py — Réponses aux questions de l'agriculteur (LECTURE SEULE).

Le LLM n'a fait que COMPRENDRE la question (sujet, cible, référence : voir schemas.Consultation).
Ici, la réponse est CONSTRUITE à partir de MongoDB, avec des phrases fixes : le modèle ne peut pas
inventer une date de délai d'attente. Si la base ne dit rien, on répond "je ne trouve rien".

Les réponses parlées sont courtes (deux phrases). Le détail complet est dans `details`,
pour l'écran.
"""
import re
import sys
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# Ajouter le répertoire Terralog au path si ce n'est pas déjà fait
# Cela permet d'importer les modules locaux quel que soit le répertoire de travail actuel
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from schemas import FUSEAU, est_vide
from stockage import en_utc, resoudre_animal, resoudre_par_nom

NB_PAROLE = 3      # nombre d'éléments lus à voix haute
NB_DETAILS = 5     # nombre d'éléments renvoyés pour l'écran

MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
        "août", "septembre", "octobre", "novembre", "décembre"]

# type_evenement (question) -> actions stockées dans interventions
GROUPES_ACTIONS = {
    "traitement": ["soin_sanitaire", "traitement_phyto"],
    "pesee": ["suivi_croissance_pesee"],
    "reproduction": ["reproduction"],
    "semis": ["semis"],
    "labour": ["labour"],
    "fertilisation": ["fertilisation"],
    "recolte": ["recolte"],
    "panne": ["panne"],
    "entretien": ["entretien_regulier", "reparation"],
    "mouvement": ["mouvement"],
    "naissance": ["naissance"],
}
LIBELLES_ACTION = {
    "soin_sanitaire": "soin", "traitement_phyto": "traitement", "suivi_croissance_pesee": "pesée",
    "reproduction": "reproduction", "semis": "semis", "labour": "labour", "recolte": "récolte",
    "fertilisation": "fertilisation", "panne": "panne", "reparation": "réparation",
    "entretien_regulier": "entretien", "plein_carburant": "plein de carburant", "mouvement": "déplacement",
    "naissance": "naissance", "sortie": "sortie", "observation": "observation", "autre": "intervention",
}
ARTICLES_CIBLE = {"animal": "l'animal", "parcelle": "la parcelle", "equipement": "la machine"}
PLURIELS_CIBLE = {"animal": "animaux", "parcelle": "parcelles", "equipement": "machines"}
COLLECTION_CIBLE = {"animal": "animaux", "parcelle": "parcelles", "equipement": "equipements"}
CHAMP_NOM = {"animal": "identification.nom_usuel", "parcelle": "identification.nom_usuel",
             "equipement": "identification.nom_usuel"}


@dataclass
class Reponse:
    parole: str
    details: list = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────────────────────
def _chemin(doc: Optional[dict], chemin: str):
    cur = doc
    for morceau in chemin.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(morceau)
    return cur


def _local(dt: datetime) -> datetime:
    return en_utc(dt).astimezone(FUSEAU)


def _jour(d) -> str:
    return f"{'1er' if d.day == 1 else d.day} {MOIS[d.month - 1]}"


def _quand(dt: datetime, maintenant: datetime) -> str:
    """aujourd'hui / hier / le 4 octobre (en heure locale de la ferme)."""
    d, ref = _local(dt).date(), _local(maintenant).date()
    ecart = (ref - d).days
    if ecart == 0:
        return "aujourd'hui"
    if ecart == 1:
        return "hier"
    return f"le {_jour(d)}"


def _jusqua(dt: datetime, maintenant: datetime) -> str:
    """jusqu'à aujourd'hui / jusqu'à demain / jusqu'au 14 octobre."""
    d, ref = _local(dt).date(), _local(maintenant).date()
    ecart = (d - ref).days
    if ecart <= 0:
        return "jusqu'à aujourd'hui"
    if ecart == 1:
        return "jusqu'à demain"
    return f"jusqu'au {_jour(d)}"


def _complement(doc: dict) -> Optional[str]:
    """Ce qui complète l'action dans une phrase : produit, mesure, détail, destination..."""
    intrants = doc.get("intrants_utilises") or []
    if intrants and intrants[0].get("valeur_textuelle"):
        return intrants[0]["valeur_textuelle"]
    details = doc.get("details") or {}
    for cle in ("dosage_ou_mesure", "description_action", "culture_recoltee", "destination"):
        if details.get(cle):
            return details[cle]
    if details.get("lieu_arrivee"):
        return f"vers {details['lieu_arrivee']}"
    return None


def _phrase_evenement(doc: dict, maintenant: datetime) -> str:
    action = doc["contexte_metier"]["action_realisee"]
    libelle = LIBELLES_ACTION.get(action, action or "intervention")
    complement = _complement(doc)
    quand = _quand(doc["horodatage"]["date_action_calculee"], maintenant)
    return f"{quand}, {libelle} {complement}" if complement else f"{quand}, {libelle}"


def _detail_evenement(doc: dict) -> dict:
    return {
        "intervention_id": str(doc["_id"]),
        "date": en_utc(doc["horodatage"]["date_action_calculee"]).isoformat(),
        "action": doc["contexte_metier"]["action_realisee"],
        "description": doc.get("description"),
        "auteur": (doc.get("auteur") or {}).get("nom"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Résolution de la cible : (doc | None, libellé, message d'erreur parlé | None)
# ──────────────────────────────────────────────────────────────────────────────
async def _resoudre_cible(db, ferme_id, cible_type: str, reference: str):
    if cible_type == "animal" and re.fullmatch(r"\d{4}", reference):
        animal = await resoudre_animal(db, ferme_id, reference)
        if animal is None:
            return None, reference, f"Je ne trouve pas la {reference} dans votre troupeau."
        return animal, f"la {reference}", None

    identifiant, statut = await resoudre_par_nom(
        db, COLLECTION_CIBLE[cible_type], ferme_id, CHAMP_NOM[cible_type], reference)
    if statut == "ambigu":
        return None, reference, f"Plusieurs {PLURIELS_CIBLE[cible_type]} ressemblent à {reference}, pouvez-vous préciser ?"
    if identifiant is None:
        return None, reference, f"Je ne connais pas {ARTICLES_CIBLE[cible_type]} {reference}."
    doc = await db[COLLECTION_CIBLE[cible_type]].find_one({"_id": identifiant})
    return doc, reference, None


def _filtre_interventions(ferme_id, cible_type: str, doc_id, type_evenement: Optional[str]) -> dict:
    filtre = {
        "ferme_id": ferme_id,
        "statut": "actif",
        "cibles.type_cible": cible_type,
        "cibles.reference_id": doc_id,
    }
    if type_evenement:
        filtre["contexte_metier.action_realisee"] = {"$in": GROUPES_ACTIONS[type_evenement]}
    return filtre


async def _derniers_evenements(db, ferme_id, cible_type, doc_id, type_evenement, n: int) -> list:
    curseur = db["interventions"].find(
        _filtre_interventions(ferme_id, cible_type, doc_id, type_evenement)
    ).sort("horodatage.date_action_calculee", -1).limit(n)
    return await curseur.to_list(n)


# ──────────────────────────────────────────────────────────────────────────────
# Réponses par sujet
# ──────────────────────────────────────────────────────────────────────────────
async def _animaux_en_delai(db, ferme_id, maintenant: datetime) -> Reponse:
    docs = await db["animaux"].find({
        "ferme_id": ferme_id,
        "cycle_de_vie.statut": "vivant",
        "fiche_sanitaire_resume.delai_attente_lait": {"$gt": maintenant},
    }).to_list(200)
    en_delai = sorted(docs, key=lambda d: en_utc(d["fiche_sanitaire_resume"]["delai_attente_lait"]))
    if not en_delai:
        return Reponse("Aucun animal n'est en délai d'attente lait pour le moment.")
    details = [{
        "boucle": d["identification"]["numero_boucle"],
        "nom": d["identification"].get("nom_usuel"),
        "delai_attente_lait_jusqu_au": en_utc(d["fiche_sanitaire_resume"]["delai_attente_lait"]).isoformat(),
    } for d in en_delai]
    lus = [f"la {d['identification']['numero_boucle']} "
           f"{_jusqua(d['fiche_sanitaire_resume']['delai_attente_lait'], maintenant)}"
           for d in en_delai[:NB_PAROLE]]
    reste = len(en_delai) - len(lus)
    suite = f", et {reste} autre{'s' if reste > 1 else ''}." if reste else "."
    return Reponse("À ne pas traire : " + ", ".join(lus) + suite, details)


def _delais_animal(animal: dict, maintenant: datetime) -> list:
    """Délais encore actifs : [("lait", dt), ("viande", dt)]."""
    actifs = []
    for produit in ("lait", "viande"):
        dt = _chemin(animal, f"fiche_sanitaire_resume.delai_attente_{produit}")
        if dt and en_utc(dt) > maintenant:
            actifs.append((produit, dt))
    return actifs


async def _delai_attente(animal: dict, libelle: str, maintenant: datetime) -> Reponse:
    actifs = _delais_animal(animal, maintenant)
    if not actifs:
        return Reponse(f"{libelle.capitalize()} n'a aucun délai d'attente en cours.")
    morceaux = [f"délai d'attente {produit} {_jusqua(dt, maintenant)}" for produit, dt in actifs]
    return Reponse(f"{libelle.capitalize()} : " + ", ".join(morceaux) + ".",
                   [{"produit": p, "jusqu_au": en_utc(dt).isoformat()} for p, dt in actifs])


async def _fiche(db, ferme_id, cible_type: str, doc: dict, libelle: str, maintenant: datetime) -> Reponse:
    parties = []
    if cible_type == "animal":
        nom = _chemin(doc, "identification.nom_usuel")
        statut = _chemin(doc, "cycle_de_vie.statut")
        parties.append(f"La {doc['identification']['numero_boucle']}" + (f", {nom}" if nom else ""))
        if statut and statut != "vivant":
            parties.append(f"statut {statut}")
        poids = _chemin(doc, "physiologie_actuelle.poids_actuel_kg")
        if poids:
            parties.append(f"{poids:g} kilos")
        lieu = _chemin(doc, "localisation_actuelle.libelle")
        if lieu:
            parties.append(f"à {lieu}")
        for produit, dt in _delais_animal(doc, maintenant):
            parties.append(f"délai {produit} {_jusqua(dt, maintenant)}")
    elif cible_type == "parcelle":
        parties.append(_chemin(doc, "identification.nom_usuel") or libelle)
        culture = _chemin(doc, "agronomie_actuelle.culture_principale")
        statut = _chemin(doc, "agronomie_actuelle.statut")
        if culture:
            parties.append(culture)
        if statut:
            parties.append(statut.replace("_", " "))
    else:
        parties.append(_chemin(doc, "identification.nom_usuel") or libelle)
        statut = _chemin(doc, "etat_actuel.statut_operationnel")
        if statut:
            parties.append(statut.replace("_", " "))

    derniers = await _derniers_evenements(db, ferme_id, cible_type, doc["_id"], None, 1)
    if derniers:
        parties.append("dernière intervention " + _phrase_evenement(derniers[0], maintenant))
    return Reponse(", ".join(parties) + ".", [_detail_evenement(d) for d in derniers])


async def _dernier_evenement(db, ferme_id, cible_type, doc, libelle, type_evenement, maintenant) -> Reponse:
    docs = await _derniers_evenements(db, ferme_id, cible_type, doc["_id"], type_evenement, 1)
    if not docs:
        precision = f" de type {type_evenement}" if type_evenement else ""
        return Reponse(f"Je ne trouve aucune intervention{precision} enregistrée pour {libelle}.")
    return Reponse(f"Dernière intervention sur {libelle} : {_phrase_evenement(docs[0], maintenant)}.",
                   [_detail_evenement(docs[0])])


async def _historique(db, ferme_id, cible_type, doc, libelle, type_evenement, maintenant) -> Reponse:
    docs = await _derniers_evenements(db, ferme_id, cible_type, doc["_id"], type_evenement, NB_DETAILS)
    if not docs:
        return Reponse(f"Je ne trouve aucune intervention enregistrée pour {libelle}.")
    lues = " ; ".join(_phrase_evenement(d, maintenant) for d in docs[:NB_PAROLE])
    return Reponse(f"Dernières interventions sur {libelle} : {lues}.", [_detail_evenement(d) for d in docs])


# ──────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ──────────────────────────────────────────────────────────────────────────────
async def repondre(db, ferme_id, question: dict, maintenant: Optional[datetime] = None) -> Reponse:
    """question : sortie du LLM (schemas.Consultation). Ne modifie jamais la base."""
    maintenant = maintenant or datetime.now(timezone.utc)
    sujet, cible_type = question.get("sujet"), question.get("cible_type")
    reference, type_evenement = question.get("reference"), question.get("type_evenement")

    if sujet == "animaux_en_delai":
        return await _animaux_en_delai(db, ferme_id, maintenant)
    if cible_type == "troupeau":
        return Reponse("Je ne sais pas encore répondre à cette question sur l'ensemble du troupeau.")
    if est_vide(sujet) and est_vide(cible_type) and est_vide(reference):
        return Reponse("Je n'ai pas compris votre question. Pouvez-vous reformuler ?")
    if est_vide(cible_type) or est_vide(reference):
        return Reponse("De quel animal, quelle parcelle ou quelle machine parlez-vous ?")

    doc, libelle, erreur = await _resoudre_cible(db, ferme_id, cible_type, reference)
    if erreur:
        return Reponse(erreur)

    sujet = sujet or "fiche"
    if sujet == "delai_attente":
        if cible_type != "animal":
            return Reponse("Les délais d'attente ne concernent que les animaux.")
        return await _delai_attente(doc, libelle, maintenant)
    if sujet == "dernier_evenement":
        return await _dernier_evenement(db, ferme_id, cible_type, doc, libelle, type_evenement, maintenant)
    if sujet == "historique":
        return await _historique(db, ferme_id, cible_type, doc, libelle, type_evenement, maintenant)
    return await _fiche(db, ferme_id, cible_type, doc, libelle, maintenant)
