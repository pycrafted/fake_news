# Détecteur de Fausses Nouvelles

Une application Streamlit qui utilise l'intelligence artificielle pour détecter les fausses nouvelles en français et en anglais.

## Fonctionnalités

- Détection automatique de la langue
- Traduction automatique vers l'anglais si nécessaire
- Interface utilisateur élégante avec le style Daktilo
- Analyse en temps réel des articles
- Affichage des scores de confiance

## Installation

1. Clonez ce dépôt
2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

Pour lancer l'application localement :
```bash
streamlit run app.py
```

## Déploiement

Cette application est déployée sur Streamlit Cloud. Vous pouvez y accéder via le lien fourni après le déploiement.

## Technologies utilisées

- Streamlit
- Transformers (Hugging Face)
- PyTorch
- Google Translate API 