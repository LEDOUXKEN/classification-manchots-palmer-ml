import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import joblib

# Configuration de la page
st.set_page_config(page_title="Classification des Manchots", page_icon="🐧", layout="wide")

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-title {color: #1E90FF; text-align: center; font-size: 2.5em;}
    .section-title {color: #FF6347; border-bottom: 2px solid #FF6347; padding-bottom: 5px;}
    .prediction {font-size: 1.2em; font-weight: bold; padding: 10px; border-radius: 5px;}
    .feature-importance {background-color: #F0F2F6; padding: 15px; border-radius: 10px;}
    .author {text-align: right; font-style: italic; color: #6A5ACD;}
    </style>
    """, unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-title">Classification des Manchots de Palmer</h1>', unsafe_allow_html=True)
st.markdown('<p class="author">Auteur : Fidèle Ledoux</p>', unsafe_allow_html=True)

# Fonction pour charger les données
@st.cache_data
def load_data():
    return pd.read_csv('palmerpenguins_extended.csv')

# Fonction pour entraîner le modèle
@st.cache_resource
def train_model(df):
    # Prétraitement
    df = df[df['species'] != 'Chinstrap']
    df = df.dropna()
    
    # Encodage des variables catégorielles
    encoder = OneHotEncoder(drop='first', sparse=False)
    categorical_cols = ['island', 'sex', 'diet', 'life_stage', 'health_metrics']
    encoded_features = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Features et target
    numerical_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'year']
    X = pd.concat([df[numerical_cols], encoded_df], axis=1)
    y = df['species']
    
    # Entraînement du modèle
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    return model, encoder, numerical_cols, categorical_cols

# Chargement des données
df = load_data()
model, encoder, numerical_cols, categorical_cols = train_model(df)

# Sidebar pour les inputs utilisateur
st.sidebar.header("Paramètres du Manchot")
st.sidebar.markdown("Renseignez les caractéristiques du manchot à classifier")

# Inputs numériques
bill_length = st.sidebar.slider("Longueur du bec (mm)", float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()), float(df['bill_length_mm'].mean()))
bill_depth = st.sidebar.slider("Profondeur du bec (mm)", float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()), float(df['bill_depth_mm'].mean()))
flipper_length = st.sidebar.slider("Longueur des nageoires (mm)", float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()), float(df['flipper_length_mm'].mean()))
body_mass = st.sidebar.slider("Masse corporelle (g)", float(df['body_mass_g'].min()), float(df['body_mass_g'].max()), float(df['body_mass_g'].mean()))
year = st.sidebar.slider("Année", int(df['year'].min()), int(df['year'].max()), int(df['year'].median()))

# Inputs catégoriels
island = st.sidebar.selectbox("Île", df['island'].unique())
sex = st.sidebar.selectbox("Sexe", df['sex'].unique())
diet = st.sidebar.selectbox("Régime alimentaire", df['diet'].unique())
life_stage = st.sidebar.selectbox("Stade de vie", df['life_stage'].unique())
health_metrics = st.sidebar.selectbox("Métriques de santé", df['health_metrics'].unique())

# Bouton de prédiction
predict_button = st.sidebar.button("Prédire l'espèce", type="primary", help="Cliquez pour faire la prédiction")

# Section principale
tab1, tab2, tab3, tab4 = st.tabs(["Prédiction", "Visualisations", "Importance des caractéristiques", "À propos"])

with tab1:
    st.markdown('<h2 class="section-title">Résultat de la prédiction</h2>', unsafe_allow_html=True)
    
    if predict_button:
        # Préparation des données d'entrée
        input_data = {
            'bill_length_mm': [bill_length],
            'bill_depth_mm': [bill_depth],
            'flipper_length_mm': [flipper_length],
            'body_mass_g': [body_mass],
            'year': [year],
            'island': [island],
            'sex': [sex],
            'diet': [diet],
            'life_stage': [life_stage],
            'health_metrics': [health_metrics]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Encodage des variables catégorielles
        encoded_input = encoder.transform(input_df[categorical_cols])
        encoded_input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(categorical_cols))
        
        # Concaténation avec les variables numériques
        final_input = pd.concat([input_df[numerical_cols].reset_index(drop=True), encoded_input_df.reset_index(drop=True)], axis=1)
        
        # Prédiction
        prediction = model.predict(final_input)
        probability = model.predict_proba(final_input)
        
        # Affichage des résultats
        if prediction[0] == 'Adelie':
            st.markdown(f'<div class="prediction" style="background-color:#FFD700;">Espèce prédite: <span style="color:#000080;">{prediction[0]}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction" style="background-color:#90EE90;">Espèce prédite: <span style="color:#006400;">{prediction[0]}</span></div>', unsafe_allow_html=True)
        
        st.write(f"Probabilité Adelie: {probability[0][0]:.2%}")
        st.write(f"Probabilité Gentoo: {probability[0][1]:.2%}")
        
        # Explication
        st.markdown("""
        **Interprétation:**
        - Le modèle a prédit l'espèce du manchot en fonction des caractéristiques fournies.
        - Les probabilités indiquent la confiance du modèle pour chaque espèce.
        """)

with tab2:
    st.markdown('<h2 class="section-title">Visualisations des données</h2>', unsafe_allow_html=True)
    
    # Sélection du type de visualisation
    viz_type = st.selectbox("Choisir une visualisation", [
        "Distribution par espèce",
        "Relation longueur/profondeur du bec",
        "Masse corporelle par île",
        "Longueur des nageoires vs masse corporelle"
    ])
    
    if viz_type == "Distribution par espèce":
        fig = px.histogram(df, x='species', color='species', 
                          title='Distribution des Manchots par Espèce',
                          labels={'species': 'Espèce'},
                          color_discrete_map={'Adelie': 'gold', 'Gentoo': 'lightgreen'})
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Relation longueur/profondeur du bec":
        fig = px.scatter(df, x='bill_length_mm', y='bill_depth_mm', color='species',
                        title='Relation entre Longueur et Profondeur du Bec',
                        labels={'bill_length_mm': 'Longueur du bec (mm)', 'bill_depth_mm': 'Profondeur du bec (mm)'},
                        color_discrete_map={'Adelie': 'gold', 'Gentoo': 'lightgreen'})
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Masse corporelle par île":
        fig = px.box(df, x='island', y='body_mass_g', color='species',
                    title='Masse Corporelle par Île et par Espèce',
                    labels={'body_mass_g': 'Masse corporelle (g)', 'island': 'Île'},
                    color_discrete_map={'Adelie': 'gold', 'Gentoo': 'lightgreen'})
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Longueur des nageoires vs masse corporelle":
        fig = px.scatter(df, x='flipper_length_mm', y='body_mass_g', color='species',
                        title='Longueur des Nageoires vs Masse Corporelle',
                        labels={'flipper_length_mm': 'Longueur des nageoires (mm)', 'body_mass_g': 'Masse corporelle (g)'},
                        color_discrete_map={'Adelie': 'gold', 'Gentoo': 'lightgreen'})
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<h2 class="section-title">Importance des caractéristiques</h2>', unsafe_allow_html=True)
    
    # Importance des features
    feature_importance = pd.DataFrame({
        'Feature': numerical_cols + list(encoder.get_feature_names_out(categorical_cols)),
        'Importance': model.coef_[0]
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    fig = px.bar(feature_importance, x='Importance', y='Feature', 
                title='Importance des Caractéristiques pour la Prédiction',
                color='Importance',
                color_continuous_scale='Bluered')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Interprétation:**
    - Les caractéristiques avec des valeurs positives contribuent davantage à la prédiction de l'espèce Gentoo.
    - Les caractéristiques avec des valeurs négatives contribuent davantage à la prédiction de l'espèce Adelie.
    """)

with tab4:
    st.markdown('<h2 class="section-title">À propos du projet</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **Projet:** Classification des Espèces de Manchots de Palmer par Régression Logistique
    
    **Objectif:** Prédire l'espèce des manchots (Adelie ou Gentoo) en utilisant des techniques de régression logistique.
    
    **Dataset:** Palmer Penguins étendu avec des variables telles que:
    - Longueur du bec
    - Masse corporelle
    - Régime alimentaire
    - Métriques de santé
    
    **Méthodologie:**
    1. Préparation des données
    2. Exploration et nettoyage
    3. Modélisation par régression logistique
    4. Évaluation des performances
    5. Déploiement avec Streamlit
    
    **Auteur:** Fidèle Ledoux
    """)
    
    st.markdown("---")
    st.markdown("""
    **Technologies utilisées:**
    - Python
    - Scikit-learn
    - Pandas
    - Streamlit
    - Plotly
    """)