# Directives du Projet

## Commandes Fréquentes
- Lancer le serveur : `uvicorn main:app --reload`
- Lancer les tests : `pytest`
- Formater / Vérifier : `ruff check .` ou `flake8`

## Architecture & Conventions
- Framework : FastAPI, MongoDB (Motor), Ollama. Le but est de réaliser plus tard une application téléphone.
- Séparation stricte : routes dans `api/`, logique métier dans `services/`, config dans `config/`.
- Les appels réseau et LLM dans les routes FastAPI doivent toujours être asynchrones (`async/await`).
- Fichiers JSON : respecter le standard JSON strict (pas de syntaxe BSON brute type `ObjectId()` ni de commentaires `//`).