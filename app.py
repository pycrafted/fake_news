import sys
import platform

# Vérification de la version Python
python_version = platform.python_version()
st.write(f"Python version: {python_version}")

import streamlit as st

# Configuration de Streamlit (doit être la première commande Streamlit)
st.set_page_config(
    page_title="Détecteur de Fausses Nouvelles",
    page_icon="🔍",
    layout="centered"
)

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import warnings
import os
from googletrans import Translator

# Configuration du style CSS personnalisé
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Daktilo&display=swap');
    
    .main {
        background-color: #f5f5f5;
    }
    
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    
    h1 {
        font-family: 'Daktilo', cursive;
        color: #2c3e50;
        text-align: center;
        font-size: 3em !important;
        margin-bottom: 0.5em;
    }
    
    .stTextArea textarea {
        font-family: 'Daktilo', cursive;
        font-size: 1.1em;
        border-radius: 10px;
        border: 2px solid #3498db;
        padding: 15px;
    }
    
    .stButton button {
        font-family: 'Daktilo', cursive;
        background-color: #3498db;
        color: white;
        border-radius: 25px;
        padding: 10px 25px;
        font-size: 1.2em;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #2980b9;
        transform: scale(1.05);
    }
    
    .prediction-box {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    
    .translation-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Désactiver les avertissements TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Initialiser le traducteur
translator = Translator()

# Charger le modèle pré-entraîné avec des paramètres plus spécifiques
@st.cache_resource
def load_model():
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        return pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu",
            framework="pt"
        )
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return None

# Titre de l'application
st.markdown("<h1>🔍 Détecteur de Fausses Nouvelles</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; font-family: Daktilo, cursive; font-size: 1.2em; color: #34495e; margin-bottom: 2em;'>
        Entrez un article de presse en français ou en anglais pour vérifier s'il s'agit d'une fausse nouvelle.
    </div>
""", unsafe_allow_html=True)

# Charger le modèle
classifier = load_model()
if classifier is None:
    st.error("Impossible de charger le modèle. Veuillez réessayer plus tard.")
    st.stop()

# Champ de saisie pour le texte
user_input = st.text_area("Texte de l'article :", height=200)

# Bouton pour analyser
if st.button("Analyser", key="analyze_button"):
    if user_input:
        try:
            # Traduire le texte en anglais si nécessaire
            try:
                # Détecter la langue
                detected_lang = translator.detect(user_input).lang
                
                # Traduire si ce n'est pas en anglais
                if detected_lang != 'en':
                    translated_text = translator.translate(user_input, dest='en').text
                    st.markdown("""
                        <div class='translation-box'>
                            <p style='font-family: Daktilo, cursive;'>
                                Le texte original était en <strong>{}</strong>. Voici la traduction en anglais :
                            </p>
                            <p style='font-family: Daktilo, cursive; font-style: italic;'>
                                {}
                            </p>
                        </div>
                    """.format(detected_lang.upper(), translated_text), unsafe_allow_html=True)
                    text_to_analyze = translated_text
                else:
                    text_to_analyze = user_input
            except Exception as e:
                st.warning("La traduction a échoué, analyse du texte original...")
                text_to_analyze = user_input

            # Faire une prédiction
            result = classifier(text_to_analyze)[0]
            label = result['label']
            score = result['score']
            
            # Afficher les détails de la prédiction
            st.markdown("""
                <div class='prediction-box'>
                    <h3 style='font-family: Daktilo, cursive; color: #2c3e50;'>Détails de l'analyse :</h3>
                    <p style='font-family: Daktilo, cursive;'>Label : {}</p>
                    <p style='font-family: Daktilo, cursive;'>Score de confiance : {:.2f}</p>
                </div>
            """.format(label, score), unsafe_allow_html=True)
            
            # Interpréter le résultat
            if label == "NEGATIVE":  # Fake news
                st.markdown("""
                    <div style='font-family: Daktilo, cursive; font-size: 1.2em; color: #e74c3c;'>
                        Cet article semble être une <strong>FAUSSE NOUVELLE</strong> (confiance : {:.2f}).
                    </div>
                    <div style='font-family: Daktilo, cursive; margin-top: 10px; color: #e74c3c;'>
                        ⚠️ Cet article contient des éléments qui suggèrent qu'il pourrait s'agir d'une fausse nouvelle.
                    </div>
                """.format(score), unsafe_allow_html=True)
            else:  # Real news
                st.markdown("""
                    <div style='font-family: Daktilo, cursive; font-size: 1.2em; color: #27ae60;'>
                        Cet article semble être <strong>VRAI</strong> (confiance : {:.2f}).
                    </div>
                    <div style='font-family: Daktilo, cursive; margin-top: 10px; color: #27ae60;'>
                        ✅ Cet article semble être une information légitime.
                    </div>
                """.format(score), unsafe_allow_html=True)
                
        except Exception as e:
            st.markdown(f"""
                <div style='font-family: Daktilo, cursive; color: #e74c3c;'>
                    <h4>Une erreur s'est produite lors de l'analyse :</h4>
                    <p>{str(e)}</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("""
                <div style='font-family: Daktilo, cursive;'>
                    <h4>Informations de débogage :</h4>
                    <p>Longueur du texte : {}</p>
                    <p>Premiers 100 caractères : {}</p>
                </div>
            """.format(len(user_input), user_input[:100]), unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='font-family: Daktilo, cursive; color: #f39c12;'>
                Veuillez entrer un texte à analyser.
            </div>
        """, unsafe_allow_html=True)