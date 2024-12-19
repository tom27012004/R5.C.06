# Recharger le fichier et filtrer les données
file_path = './Mental_Health_Music.csv'
import pandas as pd
import numpy as np

# Charger les données
data = pd.read_csv(file_path)

# Filtrer les données nécessaires
filtered_data = data.dropna(subset=["BPM", "Anxiety", "Depression", "Insomnia","Hours per day"])

# Sélectionner les colonnes essentielles pour l'optimisation
filtered_data = filtered_data[["Hours per day", "BPM", "Anxiety", "Depression", "Insomnia", "OCD"]]

# Afficher un aperçu des données filtrées
print(filtered_data.head())

# Poids pour combiner les effets mentaux
Anxiety_coef = 1
Depression_coef = 1
Insomnia_coef = 1
OCD_coef = 1

# Calculer le score combiné pour chaque personne
filtered_data["Mental_Health_Score"] = (
    100 -
    ((Anxiety_coef * filtered_data["Anxiety"] +
    Depression_coef * filtered_data["Depression"] +
    Insomnia_coef * filtered_data["Insomnia"] +
    OCD_coef * filtered_data["OCD"])/
    ((Anxiety_coef + Depression_coef + Insomnia_coef + OCD_coef)*10)
    *100)
)

# Regrouper les données par genre et calculer la moyenne des scores et le nombre de personnes par genre
grouped_data = filtered_data.groupby("Hours per day").agg(
    Mental_Health_Score_Mean=('Mental_Health_Score', 'mean'),
    Number_of_People=('Mental_Health_Score', 'size')
).reset_index()

# Trier les genres par score moyen décroissant
grouped_data = grouped_data.sort_values(by="Mental_Health_Score_Mean", ascending=False)

# Afficher les résultats
print(grouped_data)

