"""
main.py — Serveur TerraLog (FastAPI + Faster-Whisper + LLM configurable)

Un tour de dictée = transcription -> normalisation -> routage -> extraction -> validation Python.
Trois états de session :
  * rien en cours          -> nouvel enregistrement (routage puis expert)
  * relance en attente     -> l'expert complète l'enregistrement (pas de routeur)
  * en vérification        -> "oui" valide, sinon l'expert applique la correction
"""
import asyncio
import base64
import io
import json
import os
import re
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

# Ajouter le répertoire Terralog au path si ce n'est pas déjà fait
# Cela permet d'importer les modules locaux quel que soit le répertoire de travail actuel
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

import numpy as np
import ollama
from bson import ObjectId
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from gtts import gTTS
from motor.motor_asyncio import AsyncIOMotorClient

# Pour l'API Claude (cloud)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None


#client fcc (agis comme un proxy depuis les différebts modèles de LLM vers l'API OpenAI) 
_fcc_client = None

def _get_fcc_client():
    """Initialise et retourne le client OpenAI pointant vers FCC."""
    global _fcc_client
    if _fcc_client is None and OPENAI_AVAILABLE:
        _fcc_client = openai.OpenAI(
            base_url="http://localhost:8082/v1",
            api_key="fcc-dummy-key" # Clé fictive requise par le SDK, mais ignorée par FCC
        )
    return _fcc_client

import consultation
import stockage
from commandes import AIDE_PARLEE, detecter_commande, indices_de_question
from prompts import messages_expert, messages_routeur
from schemas import (
    MODELES, SCHEMAS, SCHEMA_ROUTAGE,
    appliquer_date, champs_du_modele, est_vide, fusionner, normaliser_texte,
    parser_reponse, premiere_question, purger_boucles_hallucinees, resume_vocal,
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
#Vérification de la disponibilité de l'API OpenAI pour le routage cloud
USE_CLOUD_LLM = True
# Modèle LLM local (Ollama)
LOCAL_LLM_MODEL = "llama3.1:8b"
# Modèle d'embedding local (Ollama)
LOCAL_EMBEDDING_MODEL = "nomic-embed-text"
# Modèle LLM cloud (OpenAI)
CLOUD_LLM_MODEL = "openai_api/gpt-4o-mini"
# Seuil de confiance pour le routage vectoriel
SEUIL_CONFIANCE = 0.78
# Timeout de session en minutes
SESSION_TIMEOUT_MINUTES = 5
# Dossier pour les fichiers audio temporaires
DOSSIER_AUDIO = "temp_audios"
# Sauvegarder en base de données après validation
ENREGISTRER_EN_BASE = False   # True : enregistre dans MongoDB à la validation (voir stockage.py)
# Activer le debug des prompts (peut être volumineux)
DEBUG_PROMPTS = True
# Options pour le modèle LLM local (Ollama)
OPTIONS_LLM = {"temperature": 0, "repeat_penalty": 1.0, "num_ctx": 4096}

client_mongo = AsyncIOMotorClient("mongodb://localhost:27017/")
db = client_mongo["terralog_db"]


@asynccontextmanager
async def lifespan(_app):
    if ENREGISTRER_EN_BASE:
        await stockage.initialiser_base(db)             # index (idempotent)
        rejoues = await stockage.rejouer_effets_en_attente(db)
        if rejoues:
            print(f"♻️ {rejoues} intervention(s) rattrapée(s).")
    yield


app = FastAPI(title="TerraLog Serveur IA", lifespan=lifespan)

os.makedirs(DOSSIER_AUDIO, exist_ok=True)
SESSIONS: dict = {}
VERROU_TRAITEMENT = asyncio.Lock()  # un seul tour à la fois : GPU unique, état de session cohérent


# ──────────────────────────────────────────────────────────────────────────────
# Whisper
# ──────────────────────────────────────────────────────────────────────────────
def construire_initial_prompt(parcelles=(), materiels=(), noms_animaux=()) -> str:
    """Vocabulaire de la ferme pour Whisper. À alimenter depuis la base quand elle est branchée."""
    morceaux = ["Carnet de ferme. Vêlage, boucle, mammite, anti-mammite, tank à lait, pâture, déchaumage."]
    if parcelles:
        morceaux.append(f"Parcelles : {', '.join(parcelles)}.")
    if materiels:
        morceaux.append(f"Matériel : {', '.join(materiels)}.")
    if noms_animaux:
        morceaux.append(f"Animaux : {', '.join(noms_animaux)}.")
    return " ".join(morceaux)


PROMPT_FERME = construire_initial_prompt()

print("Chargement de Faster-Whisper en VRAM, veuillez patienter...")
modele_whisper = WhisperModel("medium", device="cuda", compute_type="float16")
print("Modèle vocal prêt !")


def transcrire(chemin: str) -> str:
    segments, _ = modele_whisper.transcribe(
        chemin, language="fr", initial_prompt=PROMPT_FERME,
        vad_filter=True, beam_size=5, condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def generer_audio_base64(texte: str) -> Optional[str]:
    try:
        tts = gTTS(text=texte, lang="fr", tld="fr")
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Erreur de synthèse vocale : {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Routage vectoriel (rapide) : matrice normalisée, un seul produit matriciel
# ──────────────────────────────────────────────────────────────────────────────
CATEGORIES_ANCRES: list = []
MATRICE_ROUTAGE = None
try:
    with open("vecteurs_routage.json", "r", encoding="utf-8") as f:
        donnees_json = json.load(f)
    CATEGORIES_ANCRES = [item["categorie"] for item in donnees_json]
    MATRICE_ROUTAGE = np.array([item["vecteur"] for item in donnees_json], dtype=np.float32)
    MATRICE_ROUTAGE /= np.linalg.norm(MATRICE_ROUTAGE, axis=1, keepdims=True)
    print(f"[OK] Routeur sémantique pret : {len(CATEGORIES_ANCRES)} ancres chargees.")
except FileNotFoundError:
    print("WARNING: Fichier vecteurs_routage.json introuvable. Le routage rapide est desactive.")


def obtenir_categorie_par_vecteur(texte: str):
    if MATRICE_ROUTAGE is None:
        return None, 0.0
    rep = ollama.embeddings(model=LOCAL_EMBEDDING_MODEL, prompt=texte, keep_alive="1h")
    vecteur = np.array(rep["embedding"], dtype=np.float32)
    vecteur /= np.linalg.norm(vecteur)
    scores = MATRICE_ROUTAGE @ vecteur
    i = int(np.argmax(scores))
    return CATEGORIES_ANCRES[i], float(scores[i])


# ──────────────────────────────────────────────────────────────────────────────
# Appels LLM
# ──────────────────────────────────────────────────────────────────────────────
class ReponseIllisible(Exception):
    """Le LLM n'a pas produit de JSON valide, même après un second essai."""


def _appel_llm(messages: list, schema: dict, num_predict: int, temperature: float = 0.0) -> str:
    if USE_CLOUD_LLM and OPENAI_AVAILABLE:
        client = _get_fcc_client()
        if client is None:
            raise RuntimeError("Client FCC non initialisé. Vérifiez l'import d'OpenAI.")

        try:
            # L'API OpenAI accepte directement la liste de messages brute
            response = client.chat.completions.create(
                model=CLOUD_LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=num_predict or 1024
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Erreur lors de l'appel au proxy FCC : {e}")
            raise
    else:
        # Utiliser Ollama (local) - Code existant conservé
        rep = ollama.chat(
            model=LOCAL_LLM_MODEL, messages=messages, format=schema, keep_alive="30m",
            options={**OPTIONS_LLM, "num_predict": num_predict, "temperature": temperature},
        )
        return rep["message"]["content"]

def _extraire_json(contenu: str) -> str:
    """Filet de sécurité (modèle à raisonnement, texte autour du JSON)."""
    if "</think>" in contenu:
        contenu = contenu.split("</think>")[-1]
    m = re.search(r"\{.*\}", contenu, re.DOTALL)
    return m.group(0) if m else contenu


async def appeler_expert(categorie: str, messages: list) -> dict:
    """JSON validé par Pydantic. Un second essai à température légère si la sortie est invalide
    (à température 0, relancer donnerait exactement la même sortie)."""
    derniere_erreur = None
    for temperature in (0.0, 0.3):
        contenu = await asyncio.to_thread(_appel_llm, messages, SCHEMAS[categorie], 300, temperature)
        try:
            return parser_reponse(categorie, _extraire_json(contenu))
        except Exception as e:
            derniere_erreur = e
    raise ReponseIllisible(str(derniere_erreur))


def _memes_chiffres(a: str, b: str) -> bool:
    return re.findall(r"\d+", a) == re.findall(r"\d+", b)


async def determiner_categorie(texte: str):
    """Routage vectoriel (rapide) puis, si le score est trop faible, routeur LLM.
    Une phrase qui RESSEMBLE à une question saute le routage vectoriel : ses ancres ne connaissent que
    des enregistrements, et "quand a vêlé la 2615 ?" est très proche de "la 2615 a vêlé".
    Renvoie (categorie, texte_pour_l_expert, trace)."""
    trace = {}
    if indices_de_question(texte):
        trace["vecteur"] = "ignoré : la phrase ressemble à une question"
    else:
        categorie, score = await asyncio.to_thread(obtenir_categorie_par_vecteur, texte)
        trace["vecteur"] = {"categorie": categorie, "score": round(score, 3)}
        if categorie and score >= SEUIL_CONFIANCE:
            return categorie, texte, trace

    messages = messages_routeur(texte)
    contenu = await asyncio.to_thread(_appel_llm, messages, SCHEMA_ROUTAGE, 200)
    try:
        r = parser_reponse("routage", _extraire_json(contenu))
    except Exception:
        return "inconnue", texte, trace
    trace["routeur_llm"] = r
    # La correction phonétique du routeur n'est acceptée que si elle ne touche à aucun chiffre
    phrase = r["phrase_corrigee"] if _memes_chiffres(r["phrase_corrigee"], texte) else texte
    return r["categorie"], phrase, trace


async def mettre_a_jour(categorie: str, etat: Optional[dict], texte_expert: str,
                        texte_normalise: str, question: Optional[str] = None):
    """Un tour d'extraction, pour un nouvel enregistrement (etat=None), une relance
    ou une correction. Renvoie (donnees, a_progresse, trace). N'écrit pas dans la session."""
    messages = messages_expert(categorie, texte_expert, etat, question)
    nouvelles = await appeler_expert(categorie, messages)
    trace = {"sortie_llm": dict(nouvelles)}
    if DEBUG_PROMPTS:
        trace["messages"] = messages

    # 1. Purge des boucles inventées sur la sortie du LLM, AVANT la fusion : sinon une boucle
    #    déjà connue (recopiée par le modèle, absente de la phrase du tour) serait effacée.
    nouvelles = purger_boucles_hallucinees(nouvelles, texte_normalise)

    # 2. Fusion : une valeur connue n'est jamais écrasée par du vide
    avant = etat or {}
    donnees = fusionner(dict(avant), nouvelles)
    donnees["categorie_detectee"] = categorie

    champs = champs_du_modele(categorie)
    a_progresse = any(not est_vide(donnees.get(c)) and donnees.get(c) != avant.get(c) for c in champs)

    # 3. Date : résolue en Python. Une expression incomprise reste "à confirmer" tant
    #    qu'aucune nouvelle expression n'est donnée.
    date_ok = appliquer_date(donnees)
    if not date_ok:
        donnees["expression_date"] = None
    donnees["_date_a_confirmer"] = (not date_ok) or bool(
        avant.get("_date_a_confirmer") and est_vide(donnees.get("expression_date"))
    )
    return donnees, a_progresse, trace


# ──────────────────────────────────────────────────────────────────────────────
# Session
# ──────────────────────────────────────────────────────────────────────────────
def recuperer_ou_creer_session(session_id: str) -> dict:
    maintenant = datetime.now()
    session = SESSIONS.get(session_id)
    if session and maintenant - session["dernier_echange"] <= timedelta(minutes=SESSION_TIMEOUT_MINUTES):
        session["dernier_echange"] = maintenant
        return session
    SESSIONS[session_id] = {
        "contexte_actif": False, "en_verification": False,
        "donnees_partielles": {}, "phrases": [], "derniere_parole": None, "dernier_echange": maintenant,
    }
    return SESSIONS[session_id]


def reinitialiser_session(session: dict) -> None:
    session.update(contexte_actif=False, en_verification=False, donnees_partielles={}, phrases=[])


def conclure(session: dict, categorie: str, donnees: dict) -> None:
    """Décide : relancer (champ manquant / date à confirmer) ou demander la validation."""
    question = premiere_question(categorie, donnees)
    if question is None and donnees.get("_date_a_confirmer"):
        question = "Je n'ai pas compris la date. Quand l'avez-vous fait ?"

    if question:
        donnees["statut_donnees"] = "incomplet"
        donnees["question_relance"] = question
        session.update(contexte_actif=True, en_verification=False)
    else:
        donnees["statut_donnees"] = "a_valider"
        donnees["question_relance"] = f"Voici ce que j'ai noté : {resume_vocal(donnees)}. Tout est bon ?"
        session.update(contexte_actif=True, en_verification=True)
    session["donnees_partielles"] = donnees


# ──────────────────────────────────────────────────────────────────────────────
# Réponse API (avec voix, y compris pour les confirmations)
# ──────────────────────────────────────────────────────────────────────────────
async def reponse(texte_brut: str, trace: dict, donnees: dict, parole: Optional[str] = None) -> dict:
    publiees = {k: v for k, v in donnees.items() if not k.startswith("_")}
    sortie = {"phrase_brute": texte_brut, "trace_debug": trace, "donnees_structurees": publiees}
    a_dire = parole or publiees.get("question_relance")
    if a_dire and a_dire != "AUCUNE":
        sortie["parole"] = a_dire
        audio = await asyncio.to_thread(generer_audio_base64, a_dire)
        if audio:
            sortie["audio_base64"] = audio
    return sortie


async def valider(session: dict, texte_brut: str, trace: dict, utilisateur: str,
                  utilisateur_id: Optional[str], ferme_id: Optional[str]) -> dict:
    donnees = dict(session["donnees_partielles"])
    if "intervenant" in donnees and est_vide(donnees["intervenant"]):
        donnees["intervenant"] = utilisateur
    parole = "C'est enregistré."

    if ENREGISTRER_EN_BASE:
        contexte = stockage.Contexte(
            ferme_id=ObjectId(ferme_id),
            utilisateur_id=ObjectId(utilisateur_id) if utilisateur_id else None,
            auteur_nom=utilisateur,
            phrases=list(session["phrases"]),
            modele=MODELE_LLM,
        )
        try:
            resultat = await stockage.enregistrer(db, donnees, contexte)
        except stockage.ErreurStockage as e:
            # Rien n'est enregistré, la session reste "à valider" : l'utilisateur corrige à la voix
            return await reponse(texte_brut, {**trace, "action": "validation_refusee"},
                                 session["donnees_partielles"], parole=str(e))
        donnees["intervention_id"] = str(resultat.intervention_id)
        donnees["avertissements"] = resultat.avertissements
        phrases_avertissement = [stockage.phrase_avertissement(a) for a in resultat.avertissements]
        parole = " ".join([parole] + [p for p in phrases_avertissement if p])

    donnees["statut_donnees"] = "complet"
    donnees["question_relance"] = "AUCUNE"
    donnees["date_enregistrement"] = datetime.now().isoformat()
    reinitialiser_session(session)
    return await reponse(texte_brut, {**trace, "action": "validation"}, donnees, parole=parole)


# ──────────────────────────────────────────────────────────────────────────────
# Un tour de dictée
# ──────────────────────────────────────────────────────────────────────────────
def _publiable(donnees: dict) -> Optional[dict]:
    return {k: v for k, v in donnees.items() if not k.startswith("_")} or None


async def traiter_consultation(session: dict, texte_brut: str, texte_expert: str, trace: dict,
                               ferme_id: Optional[str], rappel: Optional[str] = None) -> dict:
    """Répond à une question (LECTURE SEULE). Le brouillon éventuel n'est jamais touché :
    `rappel` (la question en attente) est relu après la réponse."""
    question = await appeler_expert("consultation", messages_expert("consultation", texte_expert))
    trace["consultation"] = question
    if not ENREGISTRER_EN_BASE or not ferme_id:
        resultat = consultation.Reponse("Je ne peux pas consulter la ferme : la base de données n'est pas connectée.")
    else:
        try:
            resultat = await consultation.repondre(db, ObjectId(ferme_id), question)
        except Exception as e:  # une panne de base ne doit pas détruire le brouillon en cours
            print(f"Erreur de consultation : {e}")
            resultat = consultation.Reponse("Je n'ai pas pu consulter la base, réessayez dans un instant.")
    parole = f"{resultat.parole} {rappel}" if rappel else resultat.parole
    donnees = {
        "intention": "consultation",
        "reponse_parlee": resultat.parole,
        "details": resultat.details,
        "brouillon_en_cours": _publiable(session["donnees_partielles"]),
    }
    return await reponse(texte_brut, trace, donnees, parole=parole)


async def traiter_tour(session: dict, texte_brut: str, utilisateur: str,
                       utilisateur_id: Optional[str], ferme_id: Optional[str]) -> dict:
    texte = normaliser_texte(texte_brut)          # une seule normalisation, utilisée partout
    trace = {"texte_whisper": texte_brut, "texte_normalise": texte}

    if not texte:
        return await reponse(texte_brut, trace, {"erreur": "Aucune parole détectée."}, parole="Je n'ai rien entendu.")

    en_attente = session["contexte_actif"]        # une relance ou une validation attend une réponse
    etat = session["donnees_partielles"]

    # --- 1. Ordres : règles fixes, AUCUN appel au modèle (voir commandes.py) ---
    commande = detecter_commande(texte, en_attente, session["en_verification"])
    trace["commande"] = commande
    if commande == "annuler":
        reinitialiser_session(session)
        return await reponse(
            texte_brut, trace,
            {"statut": "annule", "message": "Enregistrement en cours annulé avec succès."},
            parole="D'accord, j'annule.",
        )
    if commande == "repeter":
        return await reponse(texte_brut, trace, {"commande": "repeter"},
                             parole=session.get("derniere_parole") or "Je n'ai rien à répéter.")
    if commande == "aide":
        parole = f"{AIDE_PARLEE} {etat['question_relance']}" if en_attente else AIDE_PARLEE
        return await reponse(texte_brut, trace, {"commande": "aide"}, parole=parole)
    if commande == "valider":
        return await valider(session, texte_brut, trace, utilisateur, utilisateur_id, ferme_id)

    # --- 2. Une question posée pendant un brouillon ne le touche pas ---
    routage = None
    if en_attente and indices_de_question(texte):
        routage = await determiner_categorie(texte)
        trace["routage"] = routage[2]
        if routage[0] == "consultation":
            return await traiter_consultation(session, texte_brut, routage[1], trace, ferme_id,
                                              rappel=etat["question_relance"])
        # sinon ce n'était pas une question ("quand j'ai traité la 4442, elle toussait") : on continue

    # --- 3. Vérification : tout ce qui n'est pas "oui" est une correction ---
    if session["en_verification"]:
        categorie = etat["categorie_detectee"]
        session["phrases"].append(texte_brut)     # gardées pour l'audit (donnees_ia_brutes)
        donnees, a_progresse, trace_expert = await mettre_a_jour(
            categorie, etat, texte, texte, question=etat["question_relance"])
        trace["expert"] = trace_expert
        if not a_progresse:  # état inchangé : rien n'a été compris
            return await reponse(
                texte_brut, trace, etat,
                parole="Je n'ai pas compris. Dites oui pour valider, ou dites ce qu'il faut corriger.")
        conclure(session, categorie, donnees)
        return await reponse(texte_brut, trace, donnees)

    # --- 4. Relance en attente : l'expert complète, sans passer par le routeur ---
    if en_attente:
        categorie = etat["categorie_detectee"]
        session["phrases"].append(texte_brut)
        donnees, a_progresse, trace_expert = await mettre_a_jour(
            categorie, etat, texte, texte, question=etat["question_relance"])
        trace["expert"] = trace_expert
        if a_progresse:
            conclure(session, categorie, donnees)
            return await reponse(texte_brut, trace, donnees)

        # Rien d'extrait : question sans mot interrogatif, réponse hors sujet ou changement de sujet ?
        routage = routage or await determiner_categorie(texte)
        trace["routage"] = routage[2]
        nouvelle, phrase = routage[0], routage[1]
        if nouvelle == "consultation":
            session["phrases"].pop()              # une question n'est pas une dictée
            return await traiter_consultation(session, texte_brut, phrase, trace, ferme_id,
                                              rappel=etat["question_relance"])
        if nouvelle not in MODELES or nouvelle == categorie:
            return await reponse(texte_brut, trace, etat, parole="Je n'ai pas compris. " + etat["question_relance"])
        print("Changement de sujet détecté : on abandonne l'enregistrement en cours.")
        reinitialiser_session(session)
        categorie, texte_expert = nouvelle, phrase
    else:
        # --- 5. Nouvel enregistrement, ou nouvelle question ---
        categorie, texte_expert, trace_routage = await determiner_categorie(texte)
        trace["routage"] = trace_routage
        if categorie == "consultation":
            return await traiter_consultation(session, texte_brut, texte_expert, trace, ferme_id)
        if categorie not in MODELES:
            return await reponse(
                texte_brut, trace, {"erreur": f"Catégorie non gérée : {categorie}"},
                parole="Je n'ai pas compris. Pouvez-vous reformuler ?")

    session["phrases"].append(texte_brut)
    donnees, _, trace_expert = await mettre_a_jour(categorie, None, texte_expert, texte)
    trace["expert"] = trace_expert
    trace["categorie_finale"] = categorie
    conclure(session, categorie, donnees)
    return await reponse(texte_brut, trace, donnees)


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
html_dictaphone = """
<!DOCTYPE html>
<html>
<head>
    <title>TerraLog - Test Micro</title>
    <style>
        body { font-family: sans-serif; padding: 50px; text-align: center; background-color: #f4f6f8; }
        button { padding: 15px 30px; font-size: 18px; margin: 10px; cursor: pointer; border-radius: 8px; border: none; font-weight: bold; }
        #btn-start { background-color: #4CAF50; color: white; }
        #btn-stop { background-color: #f44336; color: white; }
        button:disabled { background-color: #cccccc; cursor: not-allowed; }
        #status { margin-top: 20px; font-size: 18px; color: #555; font-weight: bold; }
        #lecteur-ia { margin-top: 20px; display: none; }
        #resultat { text-align: left; background: #282c34; color: #61dafb; padding: 20px; border-radius: 8px; display: none; margin-top: 20px; white-space: pre-wrap; font-family: monospace;}
    </style>
</head>
<body>
    <h2>🎙️ Prototype TerraLog</h2>
    <p>Cliquez pour dicter une intervention agricole.</p>
    
    <button id="btn-start">🔴 Enregistrer</button>
    <button id="btn-stop" disabled>⬛ Stopper et Envoyer</button>
    
    <p id="status">Prêt à écouter.</p>
    
    <!-- NOUVEAU : Un lecteur audio visible -->
    <audio id="lecteur-ia" controls></audio>
    
    <pre id="resultat"></pre>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        const btnStart = document.getElementById('btn-start');
        const btnStop = document.getElementById('btn-stop');
        const statusLabel = document.getElementById('status');
        const resultatDiv = document.getElementById('resultat');
        const lecteurIa = document.getElementById('lecteur-ia');

        btnStart.onclick = async () => {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = e => { audioChunks.push(e.data); };
            
            mediaRecorder.onstop = async () => {
                statusLabel.innerText = "IA en cours de traitement (Whisper + Llama)...";
                lecteurIa.style.display = "none"; // On cache le lecteur précédent
                
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                audioChunks = []; 
                
                const formData = new FormData();
                formData.append('audio_file', audioBlob, 'enregistrement.webm');
                
                try {
                    const response = await fetch('/api/v1/carnet/sync-audio', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const jsonResponse = await response.json();
                    
                    // --- NOUVEAU : GESTION DE L'AUDIO ---
                    if (jsonResponse.audio_base64) {
                        lecteurIa.src = "data:audio/mp3;base64," + jsonResponse.audio_base64;
                        lecteurIa.style.display = "inline-block";
                        
                        // On tente la lecture automatique
                        lecteurIa.play().then(() => {
                            statusLabel.innerText = "Succès ! (L'IA parle...)";
                        }).catch(e => {
                            console.warn("Lecture bloquée par le navigateur, cliquez sur Play manuellement.", e);
                            statusLabel.innerText = "Succès ! (Cliquez sur Play pour écouter la question)";
                        });
                        
                        // On supprime la variable géante de l'affichage JSON pour que ce soit lisible
                        delete jsonResponse.audio_base64; 
                    } else {
                        statusLabel.innerText = "Succès !";
                    }

                    resultatDiv.style.display = "block";
                    resultatDiv.innerText = JSON.stringify(jsonResponse, null, 2);
                    
                } catch (error) {
                    statusLabel.innerText = "Erreur de connexion au serveur.";
                    console.error(error);
                }
            };
            mediaRecorder.start();
            btnStart.disabled = true;
            btnStop.disabled = false;
            statusLabel.innerText = "🎤 Enregistrement en cours...";
        };

        btnStop.onclick = () => {
            mediaRecorder.stop();
            btnStart.disabled = false;
            btnStop.disabled = true;
        };
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def interface_test_micro():
    return html_dictaphone


EXTENSIONS_VALIDES = (".m4a", ".wav", ".mp3", ".mp4", ".webm", ".ogg")


@app.post("/api/v1/carnet/sync-audio")
async def synchroniser_audio(
    audio_file: UploadFile = File(...),
    session_id: str = Form("defaut"),
    utilisateur: str = Form("utilisateur_courant"),
    utilisateur_id: Optional[str] = Form(None),
    ferme_id: Optional[str] = Form(None),
):
    # TODO avant toute mise en production : ferme_id / utilisateur_id doivent venir d'un jeton
    # d'authentification, jamais du client (sinon n'importe qui écrit dans n'importe quelle ferme).
    if ENREGISTRER_EN_BASE and not (ferme_id and ObjectId.is_valid(ferme_id)):
        raise HTTPException(status_code=400, detail="ferme_id valide requis.")
    if utilisateur_id and not ObjectId.is_valid(utilisateur_id):
        raise HTTPException(status_code=400, detail="utilisateur_id invalide.")

    extension = os.path.splitext(audio_file.filename or "")[1].lower()
    if extension not in EXTENSIONS_VALIDES:
        raise HTTPException(status_code=400, detail="Format audio non supporté.")

    # Nom généré côté serveur : jamais le nom du client (injection de chemin "../")
    chemin = os.path.join(DOSSIER_AUDIO, f"voix_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}{extension}")
    try:
        with open(chemin, "wb") as tampon:
            shutil.copyfileobj(audio_file.file, tampon)
        print("1. Transcription Whisper en cours...")
        texte_brut = await asyncio.to_thread(transcrire, chemin)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de transcription : {e}")
    finally:
        if os.path.exists(chemin):
            os.remove(chemin)  # on ne garde pas les enregistrements vocaux

    async with VERROU_TRAITEMENT:
        session = recuperer_ou_creer_session(session_id)
        try:
            sortie = await traiter_tour(session, texte_brut, utilisateur, utilisateur_id, ferme_id)
        except ReponseIllisible as e:
            # Le LLM a échoué : on ne touche pas à la session, l'utilisateur peut répéter
            sortie = await reponse(
                texte_brut, {"erreur_technique": str(e)}, {"erreur": "Réponse de l'IA illisible."},
                parole="Je n'ai pas bien compris, pouvez-vous répéter ?")
        except Exception as e:
            print(f"Erreur inattendue : {e}")
            reinitialiser_session(session)  # on ne perpétue pas un état potentiellement corrompu
            sortie = await reponse(
                texte_brut, {"erreur_technique": str(e)}, {"erreur": f"Échec de l'analyse IA : {e}"},
                parole="Une erreur s'est produite, pouvez-vous répéter ?")
        session["derniere_parole"] = sortie.get("parole")   # pour "répète"
        return sortie
