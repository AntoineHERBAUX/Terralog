"""
schemas.py — Schémas d'extraction TerraLog (source unique : Pydantic v2)

Dépendances : pip install pydantic dateparser

Conventions :
  * "non précisé" = null (None en Python), partout, pour tous les types.
  * Le LLM ne calcule JAMAIS de date : il recopie l'expression dite
    ("hier soir") dans `expression_date`, et Python la résout (dateparser).
  * Numéro de boucle = exactement 4 chiffres collés.
  * Tous les champs sont requis dans le schéma (pour que le modèle les génère
    tous), mais nullables.
"""
import re
from datetime import datetime
from typing import Annotated, Callable, List, Literal, Optional, Tuple, get_args
from zoneinfo import ZoneInfo  # sous Windows : pip install tzdata

import dateparser
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

FORMAT_DATE = "%H:%M_%d/%m/%Y"  # format historique : HH:mm_jj/MM/aaaa
FUSEAU = ZoneInfo("Europe/Paris")  # les dates dictées sont des heures locales de la ferme

try:  # pip install text-to-num : "quarante-cinq douze" -> "45 12"
    from text_to_num import alpha2digit
except ImportError:  # la normalisation reste utilisable sans la librairie
    alpha2digit = None

# ──────────────────────────────────────────────────────────────────────────────
# Types et descriptions partagés
# ──────────────────────────────────────────────────────────────────────────────
Boucle = Annotated[str, StringConstraints(pattern=r"^[0-9]{4}$")]

DESC_BOUCLE = (
    "Numéro de boucle : exactement 4 chiffres collés (ex: '4512'), "
    "uniquement s'il est dicté. null sinon. N'invente jamais un numéro."
)
DESC_DATE = (
    "Mots de date/heure dits par l'agriculteur, recopiés tels quels "
    "(ex: 'hier soir', 'lundi à 14h', 'ce matin'). null si aucune date n'est dite. "
    "Ne calcule jamais de date."
)


class _Base(BaseModel):
    # additionalProperties: false dans le schéma JSON -> grammaire plus stricte
    model_config = ConfigDict(extra="forbid")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Routage
# ──────────────────────────────────────────────────────────────────────────────
Categorie = Literal[
    "culture",
    "elevage_intervention",
    "elevage_naissance",
    "elevage_mouvement_lot",
    "elevage_sortie",
    "materiel",
    "consultation",
    "inconnue",
]

# Source unique des définitions de catégories : réutilisable dans le prompt du routeur
CATEGORIES = {
    "culture": "champs, semis, labour, déchaumage, récolte, fertilisation, pulvérisation ou traitement des cultures",
    "elevage_intervention": "soin, vaccin, traitement, pesée ou reproduction sur un animal",
    "elevage_naissance": "vêlage, naissance ou achat/arrivée d'un animal",
    "elevage_mouvement_lot": "déplacement, transfert ou changement de lieu d'un groupe d'animaux",
    "elevage_sortie": "abattage, mort ou vente d'un animal",
    "materiel": "mécanique, panne, entretien, vidange ou plein de carburant (tracteur, tank à lait, machine)",
    "consultation": "question ou demande d'information sur la ferme (quand, quel, combien, qui, montre-moi, dis-moi, quelles vaches...), même si elle parle d'un animal : ce n'est pas un enregistrement",
    "inconnue": "tout le reste",
}
assert set(CATEGORIES) == set(get_args(Categorie)), "CATEGORIES et Categorie ont divergé"

DESCRIPTION_CATEGORIES = " ".join(f"'{k}' = {v}." for k, v in CATEGORIES.items())


class Routage(_Base):
    # L'ordre compte : le modèle génère les champs dans cet ordre.
    # Il corrige d'abord la phrase, puis classe.
    phrase_corrigee: str = Field(
        description=(
            "La phrase recopiée avec les erreurs de transcription phonétique corrigées "
            "selon le contexte agricole (ex: 'antivamite' -> 'anti-mammite'). "
            "Ne change JAMAIS les chiffres ni le sens."
        )
    )
    categorie: Categorie = Field(description=DESCRIPTION_CATEGORIES)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Cultures
# ──────────────────────────────────────────────────────────────────────────────
ACTIONS_AVEC_INTRANT = {"semis", "fertilisation", "traitement_phyto"}


class Culture(_Base):
    parcelle_concernee: Optional[str] = Field(
        description=(
            "Nom usuel du champ tel que dicté. 'POSITION_ACTUELLE' si l'agriculteur dit "
            "'ici' sans donner de nom. null si non précisé."
        )
    )
    action_realisee: Optional[
        Literal["semis", "labour", "déchamage", "recolte", "fertilisation", "traitement_phyto", "observation"]
    ] = Field(description="Opération agronomique. null si incertain.")
    intrant_ou_culture: Optional[str] = Field(
        description="Variété semée ou produit épandu/pulvérisé. null si non dit."
    )
    dose_valeur: Optional[float] = Field(
        description="Quantité par hectare, nombre seul (ex: 150). null si non dite."
    )
    dose_unite: Optional[Literal["kg/ha", "l/ha", "t/ha"]] = Field(
        description="Unité de la dose par hectare. null si non dite."
    )
    expression_date: Optional[str] = Field(description=DESC_DATE)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Élevage : intervention sur un animal
# ──────────────────────────────────────────────────────────────────────────────
class Intervention(_Base):
    numero_boucle: Optional[Boucle] = Field(description=DESC_BOUCLE)
    type_intervention: Optional[
        Literal["soin_sanitaire", "reproduction", "suivi_croissance_pesee", "autre"]
    ] = Field(description="Nature de l'acte. null si incertain.")
    produit_ou_materiel: Optional[str] = Field(
        description="Médicament, vaccin ou semence. null si non précisé."
    )
    intervenant: Optional[str] = Field(
        description=(
            "Autre personne citée qui a réalisé l'acte (ex: 'vétérinaire', 'technicien'). "
            "null si rien n'est dit (Python mettra l'utilisateur courant)."
        )
    )
    dosage_ou_mesure: Optional[str] = Field(
        description="Quantité administrée ou poids relevé, avec unité (ex: '600 kilos'). null si non précisé."
    )
    expression_date: Optional[str] = Field(description=DESC_DATE)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Élevage : naissance / arrivée
# ──────────────────────────────────────────────────────────────────────────────
class Naissance(_Base):
    numero_boucle: Optional[Boucle] = Field(description=DESC_BOUCLE)
    prenom: Optional[str] = Field(
        description="Nom donné à l'animal : un seul mot, sans phrase ni symbole. null si non dicté."
    )
    espece: Optional[Literal["bovin", "ovin", "caprin", "porcin", "autre"]] = Field(
        description=(
            "Espèce. Déduis-la du vocabulaire : veau/génisse/vêlage = bovin, agneau = ovin, "
            "chevreau = caprin, porcelet = porcin. null si aucun indice."
        )
    )
    sexe: Optional[Literal["male", "femelle"]] = Field(
        description="Sexe. Déduis-le du vocabulaire (génisse = femelle, taurillon = male). null si aucun indice."
    )
    mere: Optional[str] = Field(description="Mère de l'animal (nom ou numéro). null si non précisé.")
    localisation: Optional[str] = Field(description="Lieu de naissance. null si aucun lieu n'est cité.")
    expression_date: Optional[str] = Field(description=DESC_DATE)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Élevage : sortie
# ──────────────────────────────────────────────────────────────────────────────
class Sortie(_Base):
    numero_boucle: Optional[Boucle] = Field(description=DESC_BOUCLE)
    motif_sortie: Optional[
        Literal["abattage", "vente_elevage", "maladie", "mort_naturelle", "accident", "euthanasie", "autre"]
    ] = Field(
        description=(
            "Raison de la sortie. 'Elle est morte' seul ne suffit pas à choisir : "
            "mort_naturelle uniquement si 'mort naturelle'/'trouvée morte' est dit. null si non précisé."
        )
    )
    destination: Optional[str] = Field(description="Abattoir ou acheteur. null si non précisé.")
    expression_date: Optional[str] = Field(description=DESC_DATE)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Élevage : mouvement de lot
# ──────────────────────────────────────────────────────────────────────────────
class MouvementLot(_Base):
    numeros_boucles: List[Boucle] = Field(
        description=(
            "Liste des numéros de boucle dictés (4 chiffres chacun). "
            "Liste vide si aucun numéro n'est dicté. Jamais de termes génériques comme 'les vaches'."
        )
    )
    nom_lot: Optional[str] = Field(description="Nom propre du lot (ex: 'lot 3'). null si non cité.")
    lieu_depart: Optional[str] = Field(description="Lieu d'origine. null si non précisé.")
    lieu_arrivee: Optional[str] = Field(description="Lieu de destination. null si non précisé.")
    expression_date: Optional[str] = Field(description=DESC_DATE)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Matériel
# ──────────────────────────────────────────────────────────────────────────────
class Materiel(_Base):
    equipement: Optional[str] = Field(
        description="Équipement concerné (ex: 'tracteur John Deere', 'tank à lait'). null si non précisé."
    )
    type_operation: Optional[
        Literal["panne", "reparation", "entretien_regulier", "plein_carburant", "autre"]
    ] = Field(description="Nature de l'événement. null si non précisé.")
    description_action: Optional[str] = Field(
        description="Détails (ex: 'changement filtre à huile', 'fuite compresseur'). null si non précisé."
    )
    expression_date: Optional[str] = Field(description=DESC_DATE)


# ──────────────────────────────────────────────────────────────────────────────
# 8. Consultation (lecture seule) : le LLM comprend la question, la réponse vient de la base
# ──────────────────────────────────────────────────────────────────────────────
class Consultation(_Base):
    sujet: Optional[
        Literal["fiche", "dernier_evenement", "historique", "delai_attente", "animaux_en_delai"]
    ] = Field(
        description=(
            "Ce que l'agriculteur veut savoir. fiche = résumé d'un animal, d'une parcelle ou d'une machine. "
            "dernier_evenement = la dernière intervention. historique = les dernières interventions. "
            "delai_attente = le délai d'attente d'un animal après traitement. "
            "animaux_en_delai = la liste des animaux à ne pas traire ou en délai d'attente. null si incertain."
        )
    )
    cible_type: Optional[Literal["animal", "parcelle", "equipement", "troupeau"]] = Field(
        description="De quoi on parle. troupeau = tous les animaux. null si incertain."
    )
    reference: Optional[str] = Field(
        description=(
            "Numéro de boucle (4 chiffres), nom d'animal, de parcelle ou de machine, tel que dicté. "
            "null pour une question sur tout le troupeau ou si rien n'est cité."
        )
    )
    type_evenement: Optional[
        Literal["traitement", "pesee", "reproduction", "semis", "labour", "fertilisation",
                "recolte", "panne", "entretien", "mouvement", "naissance"]
    ] = Field(
        description="Type d'intervention demandé ('dernier traitement' = traitement). null pour tous les types."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Génération des schémas JSON pour Ollama
# ──────────────────────────────────────────────────────────────────────────────
def _generer_schema(modele: type[BaseModel]) -> dict:
    """model_json_schema() sans les 'title' (bruit inutile pour le modèle)."""

    def nettoyer(noeud):
        if isinstance(noeud, dict):
            return {
                k: nettoyer(v)
                for k, v in noeud.items()
                if not (k == "title" and isinstance(v, str))
            }
        if isinstance(noeud, list):
            return [nettoyer(x) for x in noeud]
        return noeud

    return nettoyer(modele.model_json_schema())


MODELES = {
    "culture": Culture,
    "elevage_intervention": Intervention,
    "elevage_naissance": Naissance,
    "elevage_sortie": Sortie,
    "elevage_mouvement_lot": MouvementLot,
    "materiel": Materiel,
    "consultation": Consultation,
}
SCHEMAS = {categorie: _generer_schema(m) for categorie, m in MODELES.items()}
SCHEMA_ROUTAGE = _generer_schema(Routage)

# Alias de compatibilité avec les imports actuels de main.py
SCHEMA_CULTURE = SCHEMAS["culture"]
SCHEMA_ELEVAGE = SCHEMAS["elevage_intervention"]
SCHEMA_CREATION_ANIMAL = SCHEMAS["elevage_naissance"]
SCHEMA_SORTIE_ANIMAL = SCHEMAS["elevage_sortie"]
SCHEMA_MOUVEMENT_LOT = SCHEMAS["elevage_mouvement_lot"]
SCHEMA_MATERIEL = SCHEMAS["materiel"]


def parser_reponse(categorie: str, contenu_json: str) -> dict:
    """Valide la réponse du LLM et renvoie un dict. Lève pydantic.ValidationError
    si le JSON est tronqué ou invalide (-> à traiter comme 'veuillez répéter')."""
    modele = Routage if categorie == "routage" else MODELES[categorie]
    return modele.model_validate_json(contenu_json).model_dump()


def champs_du_modele(categorie: str) -> list:
    """Noms des champs que le LLM remplit pour cette catégorie."""
    return list(MODELES[categorie].model_fields)


# ──────────────────────────────────────────────────────────────────────────────
# Dates : le LLM recopie, Python résout
# ──────────────────────────────────────────────────────────────────────────────
# L'ordre compte : "après-midi" doit être testé avant "midi".
MOMENTS_JOURNEE = {"après-midi": 15, "apres-midi": 15, "matin": 8, "midi": 12, "soir": 19, "nuit": 22}
_RE_HEURE = re.compile(r"\d{1,2}\s*(?:h|:)\s*\d{0,2}", re.IGNORECASE)


def resoudre_date(expression: Optional[str], maintenant: Optional[datetime] = None) -> Tuple[datetime, bool]:
    """Transforme 'hier soir', 'lundi à 14h'... en datetime.

    Renvoie (date, reconnue). Sans expression -> (maintenant, True).
    Expression non comprise -> (maintenant, False) : à toi de décider
    de redemander ou d'accepter.
    Une action ne peut pas être dans le futur : le résultat est plafonné à 'maintenant'.
    """
    maintenant = maintenant or datetime.now(FUSEAU).replace(tzinfo=None)  # heure locale, indépendante du serveur
    if not expression or not expression.strip():
        return maintenant, True

    texte = expression.lower().strip()

    # dateparser gère "hier", "lundi dernier", "il y a 2 jours", "hier à 14h"...
    # mais pas de façon fiable les moments de journée : on les traite nous-mêmes.
    heure_moment = None
    for mot, heure in MOMENTS_JOURNEE.items():
        motif = rf"(?<![\w-]){re.escape(mot)}(?![\w-])"
        if re.search(motif, texte):
            heure_moment = heure
            texte = re.sub(motif, " ", texte)
            break
    texte = re.sub(r"\b(?:ce|cet|cette)\b", " ", texte)
    texte = " ".join(texte.split())

    a_heure_explicite = bool(_RE_HEURE.search(texte))

    if texte:
        dt = dateparser.parse(
            texte,
            languages=["fr"],
            settings={"PREFER_DATES_FROM": "past", "RELATIVE_BASE": maintenant},
        )
        if dt is None:
            return maintenant, False
    else:
        dt = maintenant  # "ce matin" -> aujourd'hui

    if heure_moment is not None and not a_heure_explicite:
        dt = dt.replace(hour=heure_moment, minute=0, second=0, microsecond=0)

    return min(dt, maintenant), True


def appliquer_date(donnees: dict, maintenant: Optional[datetime] = None) -> bool:
    """Remplit date_action (HH:mm_jj/MM/aaaa) et date_action_iso depuis expression_date.

    Ne recalcule que si l'expression a changé depuis le dernier calcul : en session,
    "hier" ne dérive donc pas d'un tour à l'autre.
    Renvoie False si une expression a été dite mais n'a pas été comprise."""
    expression = donnees.get("expression_date")
    if donnees.get("date_action") and expression == donnees.get("_expression_resolue"):
        return True
    dt, reconnue = resoudre_date(expression, maintenant)
    donnees["date_action"] = dt.strftime(FORMAT_DATE)
    donnees["date_action_iso"] = dt.isoformat()
    donnees["_expression_resolue"] = expression
    return reconnue


# ──────────────────────────────────────────────────────────────────────────────
# Numéros de boucle : normalisation et anti-hallucination
# ──────────────────────────────────────────────────────────────────────────────
def normaliser_boucles(texte: str) -> str:
    """Recolle les numéros que Whisper découpe : '4 512' -> '4512', '45 12' -> '4512'.
    Heuristique à tester sur de vraies transcriptions."""
    texte = re.sub(r"(?<![\d/:.,-])(\d)[ .](\d{3})(?![\d/:hH-]|[.,]\d)", r"\1\2", texte)
    texte = re.sub(r"(?<![\d/:.,-])(\d{2}) (\d{2})(?![\d/:hH-]|[.,]\d|\s\d)", r"\1\2", texte)
    return texte


def normaliser_texte(texte: str) -> str:
    """Normalisation unique, à appliquer juste après Whisper : nombres en lettres -> chiffres
    (si text_to_num est installé), puis recollage des numéros de boucle."""
    if alpha2digit is not None:
        try:
            texte = alpha2digit(texte, "fr")
        except Exception:
            pass  # on garde le texte tel quel plutôt que de planter le tour
    return normaliser_boucles(texte)


def purger_boucles_hallucinees(donnees: dict, texte_normalise: str) -> dict:
    """
    Vérification assouplie : on extrait tous les chiffres de la phrase pour créer 
    une chaîne continue (ex: 'la 22-24 pèse 60kg' -> '222460').
    """
    texte_chiffres = "".join(c for c in texte_normalise if c.isdigit())

    if donnees.get("numero_boucle") and donnees["numero_boucle"] not in texte_chiffres:
        print(f"⚠️ Suspicion d'hallucination (ignorée) : l'IA a trouvé {donnees['numero_boucle']}")
        # On commente cette ligne pour ARRÊTER de détruire la donnée
        # donnees["numero_boucle"] = None

    if "numeros_boucles" in donnees:
        # On désactive aussi la suppression pour les mouvements de lots
        donnees["numeros_boucles"] = [b for b in donnees["numeros_boucles"]]

    return donnees


# ──────────────────────────────────────────────────────────────────────────────
# Complétude, fusion de session, résumé vocal
# ──────────────────────────────────────────────────────────────────────────────
def est_vide(valeur) -> bool:
    # "INCONNU"/"NON_PRECISE" : tolérés pour d'anciennes données de session
    return valeur is None or valeur == "" or valeur == [] or valeur in ("INCONNU", "NON_PRECISE")


def fusionner(anciennes: dict, nouvelles: dict) -> dict:
    """Met à jour la session sans jamais écraser une valeur connue par du vide."""
    for cle, valeur in nouvelles.items():
        if est_vide(valeur) and not est_vide(anciennes.get(cle)):
            continue
        anciennes[cle] = valeur
    return anciennes


# (champ obligatoire, question à poser, condition d'application ou None)
# La première règle non satisfaite donne la question de relance.
REGLES_RELANCE = {
    "culture": [
        ("parcelle_concernee", "Sur quelle parcelle êtes-vous intervenu ?", None),
        ("action_realisee", "Quelle action avez-vous réalisée (semis, récolte, pulvérisation...) ?", None),
        (
            "intrant_ou_culture",
            lambda d: f"Quel produit ou quelle variété avez-vous utilisé pour cette opération de {d.get('action_realisee')} ?",
            lambda d: d.get("action_realisee") in ACTIONS_AVEC_INTRANT,
        ),
        (
            "dose_valeur",
            "Quelle dose par hectare avez-vous mise ?",
            lambda d: d.get("action_realisee") in ACTIONS_AVEC_INTRANT,
        ),
    ],
    "elevage_intervention": [
        ("numero_boucle", "Quel est le numéro de boucle de l'animal concerné ?", None),
        ("type_intervention", "S'agit-il d'un soin, d'une pesée ou d'une reproduction ?", None),
        (
            "produit_ou_materiel",
            "Quel médicament ou traitement a été administré ?",
            lambda d: d.get("type_intervention") == "soin_sanitaire",
        ),
        (
            "dosage_ou_mesure",
            "Quel poids avez-vous relevé ?",
            lambda d: d.get("type_intervention") == "suivi_croissance_pesee",
        ),
    ],
    "elevage_naissance": [
        ("numero_boucle", "Quel est le numéro de boucle du nouveau-né ?", None),
        ("sexe", "S'agit-il d'un mâle ou d'une femelle ?", None),
        ("espece", "De quelle espèce animale s'agit-il ?", None),
        # La mère n'est pas demandée (le test de l'ancien main.py ne se déclenchait jamais).
        # Pour la rendre obligatoire : ("mere", "Qui est la mère ?", None)
    ],
    "elevage_sortie": [
        ("numero_boucle", "Quel est le numéro de boucle de l'animal concerné par la sortie ?", None),
        ("motif_sortie", "Quelle est la cause de la sortie (vente, abattage, mort naturelle...) ?", None),
    ],
    "materiel": [
        ("equipement", "Sur quel équipement ou machine êtes-vous intervenu ?", None),
        ("type_operation", "S'agit-il d'une panne, d'un entretien ou d'une réparation ?", None),
    ],
    "elevage_mouvement_lot": [
        (
            "numeros_boucles",
            "Quels animaux avez-vous déplacé ? Donnez les numéros de boucle ou le nom du lot.",
            lambda d: est_vide(d.get("nom_lot")),
        ),
        ("lieu_arrivee", "Vers quel lieu ou bâtiment les avez-vous transférés ?", None),
    ],
}


def premiere_question(categorie: str, donnees: dict) -> Optional[str]:
    """Question de relance pour la première info manquante, ou None si tout est là."""
    for champ, question, condition in REGLES_RELANCE.get(categorie, []):
        if condition is not None and not condition(donnees):
            continue
        if est_vide(donnees.get(champ)):
            return question(donnees) if callable(question) else question
    return None


CLES_TECHNIQUES = {
    "statut_donnees", "question_relance", "categorie_detectee", "expression_date",
    "date_action_iso", "date_enregistrement", "erreur", "parcelle_id",
}


def _date_parlee(valeur) -> str:
    try:
        d = datetime.strptime(valeur, FORMAT_DATE)
        return f"le {d:%d/%m/%Y} à {d:%H}h{d:%M}"
    except (ValueError, TypeError):
        return str(valeur)


def resume_vocal(donnees: dict) -> str:
    """Résumé lu à voix haute avant validation (valeurs non vides, sans clés techniques)."""
    morceaux = []
    for cle, valeur in donnees.items():
        if cle.startswith("_") or cle in CLES_TECHNIQUES or est_vide(valeur):
            continue
        if cle == "dose_unite" and not est_vide(donnees.get("dose_valeur")):
            continue  # déjà lue avec la valeur
        if cle == "dose_valeur":
            morceaux.append(f"{valeur:g} {donnees.get('dose_unite') or ''}".strip())
        elif cle == "date_action":
            morceaux.append(_date_parlee(valeur))
        elif isinstance(valeur, list):
            morceaux.append(", ".join(map(str, valeur)))
        else:
            morceaux.append(str(valeur))
    return ", ".join(morceaux)
