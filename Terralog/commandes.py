"""
commandes.py — Les ordres vocaux, en règles fixes (aucun appel au modèle).

Une liste fermée de mots est plus fiable qu'une IA qui interprète une phrase libre : si la phrase
n'est pas reconnue ici, elle n'est PAS traitée comme un ordre. Et comme c'est instantané,
un ordre répond tout de suite, ce qui compte quand on a les mains prises.

Ordres : valider ("oui"), annuler, répéter, aide.
Ce module détecte aussi les phrases qui RESSEMBLENT à une question : cet indice sert seulement à
envoyer la phrase au routeur LLM (qui tranche entre enregistrer et consulter), jamais à décider seul.
"""
import re
from typing import Optional

AIDE_PARLEE = (
    "Vous pouvez me dire ce que vous avez fait, par exemple : j'ai traité la 4442. "
    "Ou poser une question, par exemple : quelles vaches ne pas traire ce soir ? "
    "Dites annule pour annuler, répète pour réécouter."
)

# ── Annulation ────────────────────────────────────────────────────────────────
ANNULATION_SEULE = {"annule", "annuler", "laisse tomber", "oublie", "stop", "rien", "abandonne"}
# "rien" seul annule, mais pas dans une phrase ("il n'a rien eu")
ANNULATION_EN_PHRASE = ["annule", "annuler", "laisse tomber", "oublie", "abandonne", "stop"]

# ── Validation (seulement quand on attend un "oui") ──────────────────────────
AFFIRMATION = ["oui", "ouais", "c'est bon", "ok", "parfait", "tout est bon", "exact", "valide", "d'accord", "yes"]
OPPOSITION = ["non", "mais", "pas", "faux", "erreur", "change", "corrige", "mauvais"]

# ── Répéter / aide ───────────────────────────────────────────────────────────
REPETER = ["répète", "répète-moi", "répétez", "répétez-moi", "répéter", "repete", "redis", "redis-moi", "redites"]
AIDE_EXACTE = {"aide", "aide-moi", "à l'aide", "help"}
AIDE_MOTIFS = [
    r"qu'est-ce que je peux (?:dire|faire)",
    r"que puis-je (?:dire|faire)",
    r"quelles? (?:sont les )?commandes",
    r"comment (?:ça|ca) marche",
    r"^(?:peux-tu|pouvez-vous|tu peux|vous pouvez) m'aider$",
]

# ── Indices de question (non décisifs) ───────────────────────────────────────
DEBUTS_QUESTION = [
    "qui", "quand", "quel", "quelle", "quels", "quelles", "combien", "où", "comment", "pourquoi",
    "est-ce", "c'est quand", "c'est quel", "c'est quelle", "c'est quoi", "c'est qui",
    "montre", "montre-moi", "affiche", "affiche-moi", "dis-moi", "donne-moi", "liste",
    "y a-t-il", "reste-t-il", "faut-il", "dois-je",
]
MOTS_QUESTION_PARTOUT = [
    "quand", "combien", "quel", "quelle", "quels", "quelles", "est-ce que", "c'est quoi",
    "dernier", "dernière", "historique", "délai d'attente", "délais d'attente",
]


def normaliser(texte: str) -> str:
    """Minuscules, apostrophes droites, ponctuation retirée (tirets et apostrophes conservés)."""
    t = texte.lower().replace("’", "'")
    t = re.sub(r"[.!?,;:…«»\"]+", " ", t)
    return " ".join(t.split())


def _contient(texte: str, mots: list) -> bool:
    """Mots ENTIERS : "ok" n'est pas dans "bloqué", "pas" n'est pas dans "passage"."""
    return any(re.search(rf"(?<![\w-]){re.escape(m)}(?![\w-])", texte) for m in mots)


def est_annulation(texte: str, session_active: bool) -> bool:
    t = normaliser(texte)
    if t in ANNULATION_SEULE:
        return True
    return session_active and len(t.split()) <= 4 and _contient(t, ANNULATION_EN_PHRASE)


def est_validation(texte: str) -> bool:
    t = normaliser(texte)
    return len(t.split()) <= 6 and _contient(t, AFFIRMATION) and not _contient(t, OPPOSITION)


def est_repeter(texte: str) -> bool:
    t = normaliser(texte)
    return len(t.split()) <= 4 and _contient(t, REPETER)


def est_aide(texte: str) -> bool:
    t = normaliser(texte)
    return t in AIDE_EXACTE or any(re.search(m, t) for m in AIDE_MOTIFS)


def detecter_commande(texte: str, session_active: bool, en_verification: bool) -> Optional[str]:
    """Renvoie "annuler", "valider", "repeter", "aide" ou None. Ordre de priorité fixe."""
    if est_annulation(texte, session_active):
        return "annuler"
    if en_verification and est_validation(texte):
        return "valider"
    if est_repeter(texte):
        return "repeter"
    if est_aide(texte):
        return "aide"
    return None


def indices_de_question(texte: str) -> bool:
    """La phrase ressemble-t-elle à une question ou à une demande d'information ?
    Faux positif = un appel LLM de plus ; faux négatif = la phrase passe par le routage rapide."""
    if texte.strip().endswith("?"):
        return True
    t = normaliser(texte)
    if any(re.match(rf"{re.escape(d)}(?![\w'-])", t) for d in DEBUTS_QUESTION):
        return True
    return _contient(t, MOTS_QUESTION_PARTOUT)
