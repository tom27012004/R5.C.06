import pandas as pd
import numpy as np

# Charger les données
df = pd.read_csv("../Mental_Health_Music.csv")

# Var quantitatifs
quant = ['Age','Hours per day','BPM','Anxiety','Depression','Insomnia','OCD']

# Transformation des variables qualitatifs en variables quantitatifs
qual = ['Age', 'Hours per day', 'While working', 'Exploratory', 'BPM','Anxiety', 'Depression', 'Insomnia', 'OCD', 'Music effects']

freq = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Very frequently': 3}
df_genre = df[[col for col in df.columns if col.startswith('Frequency')]]

df_genre.replace(freq, inplace=True)
df_qual = pd.concat([df[qual], df_genre], axis=1)
df_qual['While working'] = df_qual['While working'].map({'Yes': 1, 'No': 0})
df_qual['Exploratory'] = df_qual['Exploratory'].map({'Yes': 1, 'No': 0})
df_qual['Music effects'] = df_qual['Music effects'].map({'Improve': 1, 'No effect': 0, 'Worsen': -1})

# Afficher les colonnes finales
# print(df.columns)

import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

scatter_matrix(df, alpha=0.2, figsize=(6,6), diagonal='kde', color="red")
plt.show()


# Sélectionner uniquement les variables numériques
numeric_data = df_qual.select_dtypes(include=[np.number])

# Calculer la matrice de corrélation pour un sous-ensemble
correlation_matrix = numeric_data.corr()

# Pour visualiser la corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
plt.title("Matrice de Corrélation")
plt.show()

# Scatter plot entre une variable explicative et la cible
# Rien de spécial ici
plt.scatter(df_qual["Age"], df_qual["Anxiety"])
plt.xlabel("Age")
plt.ylabel("Anxiety")
plt.title("Relation entre Variable Explicative et Cible")
plt.show()
