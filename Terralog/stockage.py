"""
stockage.py — Persistance MongoDB de TerraLog (modèle de example.json)

Principe
  * `interventions` est la SOURCE DE VÉRITÉ : un document par enregistrement validé, jamais modifié.
  * Les historiques et "états actuels" embarqués dans animaux / parcelles / equipements sont des
    copies de lecture rapide, mises à jour par `appliquer_effets`. Cette fonction est IDEMPOTENTE
    (rejouable sans doublon) : si elle échoue, l'intervention reste enregistrée, marquée
    `effets_appliques: False`, et `rejouer_effets_en_attente` rattrape.
  * Dates : datetimes BSON en UTC. Les heures dictées sont des heures locales (Europe/Paris).
  * Toute requête commence par `ferme_id` (multi-exploitation).

Dépendances : motor (pymongo/bson), rapidfuzz.
"""
import re
import sys
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

# Ajouter le répertoire Terralog au path si ce n'est pas déjà fait
# Cela permet d'importer les modules locaux quel que soit le répertoire de travail actuel
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from bson import ObjectId
from rapidfuzz import fuzz, process, utils

from schemas import FUSEAU, est_vide

SCHEMA_VERSION = 1
LIMITE_HISTORIQUE = 50   # taille de la copie de lecture rapide ; l'historique complet est dans `interventions`
SEUIL_FLOU = 75          # score minimal pour rapprocher un nom dicté d'un nom connu
ECART_AMBIGUITE = 5      # deux candidats à moins de 5 points : on ne choisit pas

HIST_ANIMAL = "fiche_sanitaire_resume.historique_evenements"
STATUT_SORTIE = {
    "abattage": "abattu", "vente_elevage": "vendu", "mort_naturelle": "mort",
    "maladie": "mort", "accident": "mort", "euthanasie": "mort", "autre": "sorti",
}


class ErreurStockage(Exception):
    """Erreur bloquante. Le message est écrit pour être lu à voix haute."""


@dataclass
class Contexte:
    ferme_id: ObjectId
    utilisateur_id: Optional[ObjectId]
    auteur_nom: str
    phrases: list = field(default_factory=list)   # phrases dictées ayant produit l'enregistrement
    modele: str = ""


@dataclass
class Resultat:
    intervention_id: ObjectId
    avertissements: list


# ──────────────────────────────────────────────────────────────────────────────
# Dates
# ──────────────────────────────────────────────────────────────────────────────
def vers_utc(dt_local_naif: datetime) -> datetime:
    return dt_local_naif.replace(tzinfo=FUSEAU).astimezone(timezone.utc)


def maintenant_utc() -> datetime:
    return datetime.now(timezone.utc)


def _utc(dt: datetime) -> datetime:
    """Mongo renvoie par défaut des datetimes naïfs (en UTC)."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ──────────────────────────────────────────────────────────────────────────────
# Index (à appeler une fois au démarrage)
# ──────────────────────────────────────────────────────────────────────────────
async def initialiser_base(db) -> None:
    await db["utilisateurs"].create_index("profil.email", unique=True)
    await db["fermes"].create_index([("localisation.coordonnees_gps", "2dsphere")])
    # Un numéro de travail à 4 chiffres n'est unique que parmi les animaux VIVANTS d'une ferme :
    # il peut être réattribué après une sortie.
    await db["animaux"].create_index(
        [("ferme_id", 1), ("identification.numero_boucle", 1)],
        unique=True, name="boucle_vivante_unique",
        partialFilterExpression={"cycle_de_vie.statut": "vivant"},
    )
    await db["animaux"].create_index([("ferme_id", 1), ("cycle_de_vie.statut", 1)])
    await db["parcelles"].create_index([("ferme_id", 1), ("identification.nom_usuel", 1)], unique=True)
    await db["parcelles"].create_index([("geometrie", "2dsphere")])
    await db["equipements"].create_index([("ferme_id", 1), ("identification.nom_usuel", 1)])
    await db["intrants"].create_index([("ferme_id", 1), ("identification.nom_commercial", 1)])
    await db["interventions"].create_index([("ferme_id", 1), ("horodatage.date_action_calculee", -1)])
    await db["interventions"].create_index(
        [("ferme_id", 1), ("cibles.type_cible", 1), ("cibles.reference_id", 1), ("horodatage.date_action_calculee", -1)])
    await db["interventions"].create_index([("ferme_id", 1), ("contexte_metier.categorie_ia", 1)])
    await db["interventions"].create_index([("statut_validation.effets_appliques", 1)])


# ──────────────────────────────────────────────────────────────────────────────
# Résolution des références (texte dicté -> _id)
# ──────────────────────────────────────────────────────────────────────────────
async def resoudre_animal(db, ferme_id, boucle: Optional[str]):
    if est_vide(boucle):
        return None
    return await db["animaux"].find_one({
        "ferme_id": ferme_id,
        "identification.numero_boucle": boucle,
        "cycle_de_vie.statut": "vivant",
    })


async def _resoudre_par_nom(db, collection: str, ferme_id, chemin_nom: str, texte: Optional[str]):
    """Rapprochement flou d'un nom dicté. Renvoie (id | None, "ok" | "inconnu" | "ambigu").
    Mieux vaut ne rien rattacher que rattacher une panne au mauvais tracteur."""
    if est_vide(texte):
        return None, "inconnu"
    docs = await db[collection].find({"ferme_id": ferme_id}, {chemin_nom: 1}).to_list(500)
    noms = {}
    for doc in docs:
        valeur = doc
        for morceau in chemin_nom.split("."):
            valeur = valeur.get(morceau) if isinstance(valeur, dict) else None
        if valeur:
            noms[valeur] = doc["_id"]
    if not noms:
        return None, "inconnu"
    meilleurs = process.extract(texte, list(noms), scorer=fuzz.WRatio, processor=utils.default_process, limit=2)
    if not meilleurs or meilleurs[0][1] < SEUIL_FLOU:
        return None, "inconnu"
    if len(meilleurs) > 1 and meilleurs[0][1] < 100 and meilleurs[0][1] - meilleurs[1][1] < ECART_AMBIGUITE:
        return None, "ambigu"
    return noms[meilleurs[0][0]], "ok"


def _noter(avertissements: list, statut: str, nature: str, texte) -> None:
    if statut == "inconnu":
        avertissements.append(f"{nature}_inconnu:{texte}")
    elif statut == "ambigu":
        avertissements.append(f"{nature}_ambigu:{texte}")


def _poids_kg(texte: Optional[str]) -> Optional[float]:
    m = re.search(r"(\d+(?:[.,]\d+)?)", texte or "")
    if not m:
        return None
    valeur = float(m.group(1).replace(",", "."))
    return valeur * 1000 if re.search(r"tonne", texte, re.IGNORECASE) else valeur


# Noms publics, utilisés aussi par consultation.py
resoudre_par_nom = _resoudre_par_nom
en_utc = _utc


# ──────────────────────────────────────────────────────────────────────────────
# Construction : une fonction par catégorie -> (cibles, intrants, details, avertissements)
# ──────────────────────────────────────────────────────────────────────────────
def _cible(type_cible: str, reference_id, texte) -> dict:
    return {"type_cible": type_cible, "reference_id": reference_id, "valeur_textuelle": texte}


async def _construire_culture(db, d, ctx, dt):
    avert, intrants, details = [], [], {}
    parcelle = d.get("parcelle_concernee")
    pid = None
    if parcelle == "POSITION_ACTUELLE":
        avert.append("position_non_resolue:")
    else:
        pid, statut = await _resoudre_par_nom(db, "parcelles", ctx.ferme_id, "identification.nom_usuel", parcelle)
        _noter(avert, statut, "parcelle", parcelle)

    action, produit = d.get("action_realisee"), d.get("intrant_ou_culture")
    if not est_vide(produit):
        if action == "recolte":
            details["culture_recoltee"] = produit      # une récolte n'utilise pas d'intrant
        else:
            iid = None
            if action in ("fertilisation", "traitement_phyto"):
                iid, statut = await _resoudre_par_nom(
                    db, "intrants", ctx.ferme_id, "identification.nom_commercial", produit)
                _noter(avert, statut, "intrant", produit)
            intrants.append({
                "intrant_id": iid, "valeur_textuelle": produit,
                "dose_hectare_appliquee": d.get("dose_valeur"), "unite_dose": d.get("dose_unite"),
            })
    return [_cible("parcelle", pid, parcelle)], intrants, details, avert


async def _construire_intervention(db, d, ctx, dt):
    avert, intrants = [], []
    boucle = d.get("numero_boucle")
    animal = await resoudre_animal(db, ctx.ferme_id, boucle)
    if animal is None:
        avert.append(f"animal_inconnu:{boucle}")

    produit = d.get("produit_ou_materiel")
    if not est_vide(produit):
        iid = None
        if d.get("type_intervention") == "soin_sanitaire":
            iid, statut = await _resoudre_par_nom(
                db, "intrants", ctx.ferme_id, "identification.nom_commercial", produit)
            _noter(avert, statut, "intrant", produit)
        intrants.append({"intrant_id": iid, "valeur_textuelle": produit})

    if d.get("type_intervention") == "suivi_croissance_pesee" and _poids_kg(d.get("dosage_ou_mesure")) is None:
        avert.append("poids_non_lu:")
    details = {"intervenant": d.get("intervenant"), "dosage_ou_mesure": d.get("dosage_ou_mesure")}
    return [_cible("animal", animal["_id"] if animal else None, boucle)], intrants, details, avert


async def _construire_naissance(db, d, ctx, dt):
    boucle = d["numero_boucle"]
    if await resoudre_animal(db, ctx.ferme_id, boucle):
        raise ErreurStockage(f"La boucle {boucle} existe déjà dans votre troupeau. Quel est le bon numéro ?")

    avert = []
    mere = d.get("mere")
    mere_id = None
    if not est_vide(mere):
        if re.fullmatch(r"\d{4}", mere):
            animal_mere = await resoudre_animal(db, ctx.ferme_id, mere)
            mere_id = animal_mere["_id"] if animal_mere else None
            statut = "ok" if mere_id else "inconnu"
        else:
            mere_id, statut = await _resoudre_par_nom(db, "animaux", ctx.ferme_id, "identification.nom_usuel", mere)
        _noter(avert, statut, "mere", mere)

    details = {c: d.get(c) for c in ("prenom", "espece", "sexe", "mere", "localisation")}
    details["mere_id"] = mere_id
    return [_cible("animal", ObjectId(), boucle)], [], details, avert


async def _construire_sortie(db, d, ctx, dt):
    avert = []
    boucle = d.get("numero_boucle")
    animal = await resoudre_animal(db, ctx.ferme_id, boucle)
    if animal is None:
        avert.append(f"animal_inconnu:{boucle}")
    elif d.get("motif_sortie") == "abattage":
        delai = (animal.get("fiche_sanitaire_resume") or {}).get("delai_attente_viande")
        if delai and _utc(delai) > dt:
            avert.append(f"delai_attente_viande:{_utc(delai).astimezone(FUSEAU):%d/%m/%Y}")
    details = {"motif_sortie": d.get("motif_sortie"), "destination": d.get("destination")}
    return [_cible("animal", animal["_id"] if animal else None, boucle)], [], details, avert


async def _construire_mouvement(db, d, ctx, dt):
    avert, cibles = [], []
    for boucle in d.get("numeros_boucles") or []:
        animal = await resoudre_animal(db, ctx.ferme_id, boucle)
        if animal is None:
            avert.append(f"animal_inconnu:{boucle}")
        cibles.append(_cible("animal", animal["_id"] if animal else None, boucle))
    if not est_vide(d.get("nom_lot")):
        cibles.append(_cible("lot", None, d["nom_lot"]))
    details = {c: d.get(c) for c in ("nom_lot", "lieu_depart", "lieu_arrivee")}
    return cibles, [], details, avert


async def _construire_materiel(db, d, ctx, dt):
    avert = []
    eid, statut = await _resoudre_par_nom(
        db, "equipements", ctx.ferme_id, "identification.nom_usuel", d.get("equipement"))
    _noter(avert, statut, "equipement", d.get("equipement"))
    return [_cible("equipement", eid, d.get("equipement"))], [], {"description_action": d.get("description_action")}, avert


CONSTRUCTEURS = {
    "culture": _construire_culture,
    "elevage_intervention": _construire_intervention,
    "elevage_naissance": _construire_naissance,
    "elevage_sortie": _construire_sortie,
    "elevage_mouvement_lot": _construire_mouvement,
    "materiel": _construire_materiel,
}


def _action(categorie: str, d: dict) -> Optional[str]:
    return {
        "culture": d.get("action_realisee"),
        "elevage_intervention": d.get("type_intervention"),
        "elevage_naissance": "naissance",
        "elevage_sortie": "sortie",
        "elevage_mouvement_lot": "mouvement",
        "materiel": d.get("type_operation"),
    }[categorie]


def _description(categorie: str, d: dict) -> str:
    dose = f"{d['dose_valeur']:g} {d.get('dose_unite') or ''}".strip() if not est_vide(d.get("dose_valeur")) else None
    parties = {
        "culture": [d.get("action_realisee"), d.get("intrant_ou_culture"), dose, "sur", d.get("parcelle_concernee")],
        "elevage_intervention": [d.get("type_intervention"), d.get("produit_ou_materiel"),
                                 d.get("dosage_ou_mesure"), d.get("numero_boucle")],
        "elevage_naissance": ["naissance", d.get("espece"), d.get("sexe"), d.get("numero_boucle")],
        "elevage_sortie": ["sortie", d.get("motif_sortie"), d.get("destination"), d.get("numero_boucle")],
        "elevage_mouvement_lot": ["mouvement", d.get("lieu_depart"), "->", d.get("lieu_arrivee")],
        "materiel": [d.get("type_operation"), d.get("equipement"), d.get("description_action")],
    }[categorie]
    if categorie == "culture" and est_vide(d.get("parcelle_concernee")):
        parties = parties[:-2]
    return " ".join(str(p) for p in parties if not est_vide(p))


# ──────────────────────────────────────────────────────────────────────────────
# Enregistrement
# ──────────────────────────────────────────────────────────────────────────────
async def enregistrer(db, donnees: dict, ctx: Contexte) -> Resultat:
    """Enregistre une intervention validée puis met à jour les copies de lecture rapide."""
    categorie = donnees["categorie_detectee"]
    dt = vers_utc(datetime.fromisoformat(donnees["date_action_iso"]))
    cibles, intrants, details, avertissements = await CONSTRUCTEURS[categorie](db, donnees, ctx, dt)

    doc = {
        "_id": ObjectId(),
        "schema_version": SCHEMA_VERSION,
        "ferme_id": ctx.ferme_id,
        "auteur": {"utilisateur_id": ctx.utilisateur_id, "nom": ctx.auteur_nom},
        "horodatage": {
            "date_enregistrement": maintenant_utc(),
            "date_action_calculee": dt,
            "expression_date": donnees.get("expression_date"),
        },
        "contexte_metier": {"categorie_ia": categorie, "action_realisee": _action(categorie, donnees)},
        "description": _description(categorie, donnees),
        "cibles": cibles,
        "intrants_utilises": intrants,
        "details": details,
        "donnees_ia_brutes": {
            "phrases_brutes": ctx.phrases,
            "modele_utilise": ctx.modele,
            "donnees_extraites": {k: v for k, v in donnees.items()
                                  if not k.startswith("_") and k != "question_relance"},
        },
        "statut_validation": {"etat": "complet", "mode_validation": "vocale_utilisateur", "effets_appliques": False},
        "statut": "actif",
    }
    await db["interventions"].insert_one(doc)

    try:
        await appliquer_effets(db, doc)
        await db["interventions"].update_one({"_id": doc["_id"]}, {"$set": {"statut_validation.effets_appliques": True}})
    except Exception as e:  # l'intervention est sauvegardée : on rattrapera avec rejouer_effets_en_attente
        print(f"⚠️ Effets différés pour {doc['_id']} : {e}")
        avertissements.append("effets_differes:")
    return Resultat(doc["_id"], avertissements)


# ──────────────────────────────────────────────────────────────────────────────
# Effets sur les copies de lecture rapide (idempotents)
# ──────────────────────────────────────────────────────────────────────────────
async def _pousser(collection, doc_id, chemin: str, evenement: dict) -> None:
    """Ajoute l'événement à l'historique, une seule fois, en gardant les LIMITE_HISTORIQUE derniers."""
    await collection.update_one(
        {"_id": doc_id, f"{chemin}.intervention_id": {"$ne": evenement["intervention_id"]}},
        {"$push": {chemin: {"$each": [evenement], "$slice": -LIMITE_HISTORIQUE}},
         "$set": {"metadonnees.derniere_mise_a_jour": maintenant_utc()}},
    )


async def _maj_etat(collection, doc_id, champ_date: str, valeurs: dict, dt: datetime) -> None:
    """Met à jour un "état actuel" seulement si cet événement est plus récent que le dernier appliqué
    (une intervention dictée en retard ne doit pas écraser un état plus récent)."""
    await collection.update_one(
        {"_id": doc_id, "$or": [{champ_date: None}, {champ_date: {"$lte": dt}}]},
        {"$set": {**valeurs, champ_date: dt}},
    )


async def appliquer_effets(db, doc: dict) -> None:
    categorie = doc["contexte_metier"]["categorie_ia"]
    action = doc["contexte_metier"]["action_realisee"]
    dt = doc["horodatage"]["date_action_calculee"]
    details = doc["details"]
    evenement = {"date_evenement": dt, "categorie": categorie, "action": action,
                 "description": doc["description"], "intervention_id": doc["_id"]}
    animaux = db["animaux"]
    ids_animaux = [c["reference_id"] for c in doc["cibles"] if c["type_cible"] == "animal" and c["reference_id"]]

    if categorie == "culture":
        parcelles = db["parcelles"]
        for cible in doc["cibles"]:
            pid = cible["reference_id"]
            if cible["type_cible"] != "parcelle" or not pid:
                continue
            await _pousser(parcelles, pid, "historique_interventions", evenement)
            if action == "semis":
                valeurs = {"agronomie_actuelle.date_semis": dt, "agronomie_actuelle.statut": "en_croissance"}
                if doc["intrants_utilises"]:
                    valeurs["agronomie_actuelle.culture_principale"] = doc["intrants_utilises"][0]["valeur_textuelle"]
                await _maj_etat(parcelles, pid, "agronomie_actuelle.date_statut", valeurs, dt)
            elif action == "recolte":
                await _maj_etat(parcelles, pid, "agronomie_actuelle.date_statut",
                                {"agronomie_actuelle.statut": "recolte"}, dt)

    elif categorie == "elevage_naissance":
        cible = doc["cibles"][0]
        localisation = details.get("localisation")
        animal = {
            "schema_version": SCHEMA_VERSION,
            "ferme_id": doc["ferme_id"],
            "identification": {
                "numero_boucle": cible["valeur_textuelle"], "numero_national_complet": None,
                "nom_usuel": details.get("prenom"), "espece": details.get("espece"),
                "race": None, "sexe": details.get("sexe"),
            },
            "genealogie": {"mere": details.get("mere"), "mere_id": details.get("mere_id"), "pere": None},
            "cycle_de_vie": {"statut": "vivant", "date_entree": dt, "motif_entree": "naissance",
                             "date_sortie": None, "motif_sortie": None, "destination_sortie": None},
            "physiologie_actuelle": {"statut_reproduction": None, "stade_lactation": None,
                                     "poids_actuel_kg": None, "date_derniere_pesee": None},
            "fiche_sanitaire_resume": {"delai_attente_lait": None, "delai_attente_viande": None,
                                       "historique_evenements": [evenement]},
            "localisation_actuelle": {"libelle": localisation, "depuis": dt if localisation else None},
            "metadonnees": {"date_creation": maintenant_utc(), "derniere_mise_a_jour": maintenant_utc()},
        }
        # upsert + $setOnInsert : rejouer ne crée pas de doublon
        await animaux.update_one({"_id": cible["reference_id"]}, {"$setOnInsert": animal}, upsert=True)

    elif categorie in ("elevage_intervention", "elevage_sortie", "elevage_mouvement_lot"):
        for aid in ids_animaux:
            await _pousser(animaux, aid, HIST_ANIMAL, evenement)

            if action == "suivi_croissance_pesee":
                poids = _poids_kg(details.get("dosage_ou_mesure"))
                if poids is not None:
                    await _maj_etat(animaux, aid, "physiologie_actuelle.date_derniere_pesee",
                                    {"physiologie_actuelle.poids_actuel_kg": poids}, dt)

            elif action == "soin_sanitaire":
                await _appliquer_delais_attente(db, aid, doc, dt)

            elif categorie == "elevage_sortie":
                await animaux.update_one({"_id": aid}, {"$set": {
                    "cycle_de_vie.statut": STATUT_SORTIE.get(details.get("motif_sortie"), "sorti"),
                    "cycle_de_vie.date_sortie": dt,
                    "cycle_de_vie.motif_sortie": details.get("motif_sortie"),
                    "cycle_de_vie.destination_sortie": details.get("destination"),
                }})

            elif categorie == "elevage_mouvement_lot" and details.get("lieu_arrivee"):
                await _maj_etat(animaux, aid, "localisation_actuelle.depuis",
                                {"localisation_actuelle.libelle": details["lieu_arrivee"]}, dt)

    elif categorie == "materiel":
        equipements = db["equipements"]
        for cible in doc["cibles"]:
            eid = cible["reference_id"]
            if cible["type_cible"] != "equipement" or not eid:
                continue
            await _pousser(equipements, eid, "historique_interventions", evenement)
            if action == "panne":
                await _maj_etat(equipements, eid, "etat_actuel.date_changement_statut",
                                {"etat_actuel.statut_operationnel": "en_panne"}, dt)
            elif action == "reparation":
                await _maj_etat(equipements, eid, "etat_actuel.date_changement_statut",
                                {"etat_actuel.statut_operationnel": "operationnel"}, dt)
            elif action == "entretien_regulier":
                await _maj_etat(equipements, eid, "maintenance_et_conformite.date_dernier_entretien", {}, dt)


async def _appliquer_delais_attente(db, animal_id, doc: dict, dt: datetime) -> None:
    """Délais d'attente lait/viande d'après la carence du produit. `$max` : on garde le plus tardif.
    Le "statut global" (en traitement ou non) se déduit à la lecture : now < délai."""
    iid = next((i["intrant_id"] for i in doc["intrants_utilises"] if i.get("intrant_id")), None)
    if iid is None:
        return
    intrant = await db["intrants"].find_one({"_id": iid}, {"reglementation_et_securite": 1})
    regl = (intrant or {}).get("reglementation_et_securite") or {}
    maxi = {}
    if regl.get("carence_lait_jours"):
        maxi["fiche_sanitaire_resume.delai_attente_lait"] = dt + timedelta(days=regl["carence_lait_jours"])
    if regl.get("carence_viande_jours"):
        maxi["fiche_sanitaire_resume.delai_attente_viande"] = dt + timedelta(days=regl["carence_viande_jours"])
    if maxi:
        await db["animaux"].update_one({"_id": animal_id}, {"$max": maxi})


async def rejouer_effets_en_attente(db, limite: int = 100) -> int:
    """À lancer au démarrage ou périodiquement : rattrape les interventions dont les effets ont échoué."""
    docs = await db["interventions"].find({"statut_validation.effets_appliques": False}).to_list(limite)
    rejoues = 0
    for doc in docs:
        try:
            await appliquer_effets(db, doc)
            await db["interventions"].update_one(
                {"_id": doc["_id"]}, {"$set": {"statut_validation.effets_appliques": True}})
            rejoues += 1
        except Exception as e:
            print(f"⚠️ Rejeu impossible pour {doc['_id']} : {e}")
    return rejoues


# ──────────────────────────────────────────────────────────────────────────────
# Avertissements parlés ("code:valeur" -> phrase)
# ──────────────────────────────────────────────────────────────────────────────
def phrase_avertissement(code: str) -> str:
    nature, _, valeur = code.partition(":")
    return {
        "animal_inconnu": f"Attention, je ne trouve pas la {valeur} dans votre troupeau.",
        "parcelle_inconnu": f"Attention, je ne connais pas la parcelle {valeur}.",
        "parcelle_ambigu": f"Attention, plusieurs parcelles ressemblent à {valeur}.",
        "intrant_inconnu": f"Attention, je ne connais pas le produit {valeur}.",
        "intrant_ambigu": f"Attention, plusieurs produits ressemblent à {valeur}.",
        "equipement_inconnu": f"Attention, je ne connais pas l'équipement {valeur}.",
        "equipement_ambigu": f"Attention, plusieurs équipements ressemblent à {valeur}.",
        "mere_inconnu": f"Attention, je ne connais pas la mère {valeur}.",
        "delai_attente_viande": f"Attention, cet animal est en délai d'attente viande jusqu'au {valeur}.",
        "poids_non_lu": "Je n'ai pas pu lire le poids, il n'est pas mis à jour sur la fiche.",
        "position_non_resolue": "Je ne sais pas encore retrouver la parcelle depuis votre position.",
        "effets_differes": "La fiche sera mise à jour un peu plus tard.",
    }.get(nature, "")
