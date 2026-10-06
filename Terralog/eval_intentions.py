"""
eval_intentions.py — Mesure, sur le VRAI modèle (Ollama doit tourner), si le routeur distingue
une affirmation (on enregistre) d'une question (on consulte), puis si l'extraction des
questions est correcte champ par champ.

    python eval_intentions.py

Ces phrases sont distinctes des exemples few-shot de prompts.py. Remplace-les et complète-les
par de vraies dictées dès que tu en as : c'est ton jeu de test, pas un exemple.

Ce que le script NE teste pas : le routage vectoriel (ancres). Il teste l'étape LLM, celle qui
tranche quand une phrase ressemble à une question. Les consultations SANS indice de question
(voir le rapport) passeraient d'abord par les ancres : surveille-les.
"""
import json
import os
import sys

# Ajouter le répertoire Terralog au path si ce n'est pas déjà fait
# Cela permet d'importer les modules locaux quel que soit le répertoire de travail actuel
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

import ollama

# Pour l'API Claude (cloud)
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

from commandes import indices_de_question
from prompts import messages_expert, messages_routeur
from schemas import SCHEMA_ROUTAGE, SCHEMAS, parser_reponse

# Configuration du provider LLM pour l'évaluation
USE_CLOUD_LLM = False  # True pour utiliser l'API Claude (cloud), False pour Ollama (local)
CLOUD_LLM_MODEL = "claude-3-5-sonnet-20241022"  # Modèle à utiliser en cloud
LOCAL_LLM_MODEL = "llama3.1:8b"  # Modèle à utiliser en local avec Ollama
OPTIONS = {"temperature": 0, "repeat_penalty": 1.0, "num_ctx": 4096}

# Client Anthropic (initialisé si nécessaire et si la clé est disponible)
_anthropic_client = None


def _get_anthropic_client():
    """Initialise et retourne le client Anthropic si disponible et configuré."""
    global _anthropic_client
    if _anthropic_client is None and ANTHROPIC_AVAILABLE:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _appel_llm(messages: list, schema: dict, num_predict: int, temperature: float = 0.0) -> str:
    """Fonction d'appel LLM unifiée qui bascule entre local et cloud selon la configuration."""
    if USE_CLOUD_LLM and ANTHROPIC_AVAILABLE:
        # Utiliser l'API Claude (cloud)
        client = _get_anthropic_client()
        if client is None:
            raise RuntimeError("API Claude configurée mais client non initialisé. Vérifiez ANTHROPIC_API_KEY.")

        # Convertir les messages au format attendu par Anthropic
        system_messages = [m["content"] for m in messages if m["role"] == "system"]
        chat_messages = [m for m in messages if m["role"] in ["user", "assistant"]]

        system_prompt = "\n".join(system_messages) if system_messages else ""

        # Pour simplifier, on prend juste le dernier message utilisateur si c'est une requête simple
        user_content = ""
        for m in reversed(chat_messages):
            if m["role"] == "user":
                user_content = m["content"]
                break

        # Appel à l'API Claude
        try:
            response = client.messages.create(
                model=CLOUD_LLM_MODEL,
                max_tokens=num_predict or 1024,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}]
            )
            return response.content[0].text if response.content else ""
        except Exception as e:
            print(f"Erreur lors de l'appel à l'API Claude : {e}")
            raise
    else:
        # Utiliser Ollama (local)
        rep = ollama.chat(
            model=LOCAL_LLM_MODEL, messages=messages, format=schema,
            options={**OPTIONS, "num_predict": num_predict}
        )
        return rep["message"]["content"]

# (phrase dictée, catégorie attendue du routeur)
ENREGISTREMENTS = [
    ("J'ai pesé la 7204, 590 kilos.", "elevage_intervention"),
    ("J'ai vacciné la 8319 ce matin.", "elevage_intervention"),
    ("Quand j'ai traité la 8841, elle toussait.", "elevage_intervention"),     # mot interrogatif, mais affirmation
    ("J'ai désherbé les Peupliers hier soir.", "culture"),
    ("J'ai fait le dernier passage de herse sur le Grand Champ.", "culture"),   # "dernier", mais affirmation
    ("J'ai épandu de l'azote aux Grandes Terres.", "culture"),
    ("La 6033 a vêlé cette nuit.", "elevage_naissance"),
    ("J'ai vendu la 4096 à Lefebvre.", "elevage_sortie"),
    ("J'ai déplacé la 3358 et la 3359 dans le pré du haut.", "elevage_mouvement_lot"),
    ("Le tracteur John Deere a une fuite d'huile.", "materiel"),
    ("J'ai fait le plein du Fendt.", "materiel"),
    ("Il va pleuvoir toute la semaine.", "inconnue"),
]

# (phrase dictée, champs attendus de la consultation)
CONSULTATIONS = [
    ("Est-ce que la 5090 est encore en délai d'attente ?",
     {"sujet": "delai_attente", "cible_type": "animal", "reference": "5090", "type_evenement": None}),
    ("Dis-moi quand j'ai pesé la 7204 la dernière fois.",
     {"sujet": "dernier_evenement", "cible_type": "animal", "reference": "7204", "type_evenement": "pesee"}),
    ("Quelle est la dernière chose faite aux Peupliers ?",
     {"sujet": "dernier_evenement", "cible_type": "parcelle", "reference": "Peupliers", "type_evenement": None}),
    ("Quelles vaches sont encore en délai d'attente ?",
     {"sujet": "animaux_en_delai", "cible_type": "troupeau", "reference": None, "type_evenement": None}),
    ("Donne-moi l'historique de la 8319.",
     {"sujet": "historique", "cible_type": "animal", "reference": "8319", "type_evenement": None}),
    ("Fiche de Marguerite.",                                                  # aucun indice de question
     {"sujet": "fiche", "cible_type": "animal", "reference": "Marguerite", "type_evenement": None}),
    ("Qu'est-ce qui s'est passé sur le tracteur John Deere ?",
     {"sujet": "historique", "cible_type": "equipement", "reference": "John Deere", "type_evenement": None}),
    ("Le tank à lait, c'est quand son dernier entretien ?",
     {"sujet": "dernier_evenement", "cible_type": "equipement", "reference": "tank à lait", "type_evenement": "entretien"}),
    ("La 4096 peut être traite quand ?",
     {"sujet": "delai_attente", "cible_type": "animal", "reference": "4096", "type_evenement": None}),
    ("Dernier semis sur le Grand Champ ?",
     {"sujet": "dernier_evenement", "cible_type": "parcelle", "reference": "Grand Champ", "type_evenement": "semis"}),
    ("Où en est la 3358 ?",
     {"sujet": "fiche", "cible_type": "animal", "reference": "3358", "type_evenement": None}),
    ("Qui est en traitement en ce moment ?",
     {"sujet": "animaux_en_delai", "cible_type": "troupeau", "reference": None, "type_evenement": None}),
]


def appel(chat, messages, schema, num_predict):
    # Le paramètre 'chat' est conservé pour compatibilité mais ignoré
    # car nous utilisons maintenant notre fonction unifiée _appel_llm
    return _appel_llm(messages, schema, num_predict)


def router(chat, phrase: str) -> str:
    try:
        return parser_reponse("routage", appel(chat, messages_routeur(phrase), SCHEMA_ROUTAGE, 200))["categorie"]
    except Exception:
        return "illisible"


def champ_ok(attendu, obtenu) -> bool:
    if attendu is None:
        return obtenu is None
    if obtenu is None:
        return False
    return str(attendu).lower() in str(obtenu).lower()      # "Peupliers" accepté dans "les Peupliers"


def evaluer(chat=ollama.chat) -> dict:
    # ── 1. Enregistrer ou consulter ? ───────────────────────────────────────────
    matrice = {("enregistrement", "enregistrement"): 0, ("enregistrement", "consultation"): 0,
               ("consultation", "enregistrement"): 0, ("consultation", "consultation"): 0}
    erreurs, sans_indice = [], []
    cas = [(p, "enregistrement", c) for p, c in ENREGISTREMENTS] + [(p, "consultation", "consultation") for p, _ in CONSULTATIONS]
    for phrase, attendu, categorie_attendue in cas:
        obtenu_cat = router(chat, phrase)
        obtenu = "consultation" if obtenu_cat == "consultation" else "enregistrement"
        matrice[(attendu, obtenu)] += 1
        if obtenu != attendu:
            erreurs.append((phrase, categorie_attendue, obtenu_cat))
        elif attendu == "enregistrement" and obtenu_cat != categorie_attendue and obtenu_cat != "illisible":
            erreurs.append((phrase, categorie_attendue, obtenu_cat))   # bon côté, mauvaise catégorie
        if attendu == "consultation" and not indices_de_question(phrase):
            sans_indice.append(phrase)

    # ── 2. Extraction des consultations, champ par champ ────────────────────────
    champs = ["sujet", "cible_type", "reference", "type_evenement"]
    juste = {c: 0 for c in champs}
    fautes = []
    for phrase, attendu in CONSULTATIONS:
        try:
            sortie = parser_reponse("consultation", appel(chat, messages_expert("consultation", phrase), SCHEMAS["consultation"], 300))
        except Exception:
            sortie = {c: "ILLISIBLE" for c in champs}
        for c in champs:
            if champ_ok(attendu[c], sortie.get(c)):
                juste[c] += 1
            else:
                fautes.append((phrase, c, attendu[c], sortie.get(c)))
    return {"matrice": matrice, "erreurs": erreurs, "sans_indice": sans_indice,
            "juste": juste, "total": len(CONSULTATIONS), "fautes": fautes}


def afficher(r: dict) -> None:
    m = r["matrice"]
    print("\n=== Enregistrer ou consulter ? (lignes = attendu, colonnes = obtenu) ===")
    print(f"{'':>16} | {'enregistrement':>14} | {'consultation':>12}")
    for attendu in ("enregistrement", "consultation"):
        print(f"{attendu:>16} | {m[(attendu, 'enregistrement')]:>14} | {m[(attendu, 'consultation')]:>12}")
    graves = m[("enregistrement", "consultation")] + m[("consultation", "enregistrement")]
    print(f"Confusions : {graves}  (affirmation prise pour une question = rien n'est enregistré ; "
          f"question prise pour une affirmation = un faux brouillon)")
    if r["erreurs"]:
        print("\nÀ regarder :")
        for phrase, attendu, obtenu in r["erreurs"]:
            print(f"  - {phrase!r}\n      attendu {attendu}, obtenu {obtenu}")

    print("\n=== Extraction des consultations (champ par champ) ===")
    for champ, n in r["juste"].items():
        print(f"  {champ:>15} : {n}/{r['total']}")
    for phrase, champ, attendu, obtenu in r["fautes"]:
        print(f"  - {phrase!r} : {champ} attendu {attendu!r}, obtenu {obtenu!r}")

    if r["sans_indice"]:
        print("\n=== Consultations SANS indice de question ===")
        print("En conditions réelles, ces phrases passent d'abord par les ancres vectorielles et risquent")
        print("d'être prises pour un enregistrement. Ajoute des ancres 'consultation' ou des mots à commandes.py :")
        for phrase in r["sans_indice"]:
            print(f"  - {phrase!r}")


if __name__ == "__main__":
    afficher(evaluer())
