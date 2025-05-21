# Classification des Espèces de Manchots de Palmer | Régression Logistique 

##  Introduction
Ce projet vise à prédire l’espèce des manchots Adelie et Gentoo à partir de **données biologiques et environnementales**, en utilisant des **techniques de régression logistique**.

##  Structure du Projet
- `data/` : Jeu de données prétraité  
- `notebooks/` : Analyses exploratoires et entraînement des modèles  
- `models/` : Sauvegarde du meilleur modèle  
- `scripts/` : Code de prétraitement et prédiction  

##  Technologies Utilisées
- **Data Processing** : `Pandas`, `NumPy`, `OneHotEncoder`
- **Modélisation** : `Logistic Regression`, `Scikit-learn`
- **Évaluation** : `Confusion Matrix`, `Accuracy Score`, `F1-Score`
- **Visualisation** : `Matplotlib`, `Seaborn`, `Heatmaps`
- <img width="536" alt="image" src="https://github.com/user-attachments/assets/bc106375-81b2-4359-a8ed-c71684d0bcd7" />
<img width="539" alt="image" src="https://github.com/user-attachments/assets/a87f8cd4-fb31-492f-bad6-c1a0fbc7c00b" />
<img width="546" alt="image" src="https://github.com/user-attachments/assets/b7e77ca7-58ef-47ae-8fec-35b9f6cd06d1" />
<img width="552" alt="image" src="https://github.com/user-attachments/assets/bdd182df-f599-4f48-aec6-6fcf5b52036f" />
<img width="519" alt="image" src="https://github.com/user-attachments/assets/a784b9ca-afaa-4cd8-b964-12b106b5ed96" />
<img width="281" alt="image" src="https://github.com/user-attachments/assets/00195664-8c89-4d1d-8504-d3ad34111842" />

##  Résultats
La **régression logistique** atteint une **précision de 94%** dans la classification des espèces de manchots.

##  Comment Exécuter
1. Installez les dépendances : `pip install -r requirements.txt`  
2. Lancez l'entraînement du modèle : `python train_model.py`  
3. Faites des prédictions : `python predict.py --bill_length 45 --bill_depth 18 --flipper_length 220 --body_mass 5000`

##  Auteur
 **Fidèle-Ledoux** - Passionné de Data Science et Modélisation appliquée à l'écologie.

🔗 [LinkedIn](https://www.linkedin.com/in/fidele-ledoux) | [Portfolio](#)
