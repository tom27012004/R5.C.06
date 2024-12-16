import warnings
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
correlation_matrix = df[quant].corr()

# Pour visualiser la corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Matrice de Corrélation 1")
plt.show()

# Calculer la matrice de corrélation pour un sous-ensemble
correlation_matrix = numeric_data.corr()

# Pour visualiser la corrélation avec plus d'éléments
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
plt.title("Matrice de Corrélation  2")
plt.show()


plt.figure(figsize=(10, 8), layout="constrained")
plt.subplot(3,3,1)
sns.histplot(df['Age'], kde=True, color='#144552')
plt.subplot(3,3,2)
sns.histplot(df['Hours per day'], kde=True, color='#1B3A4B')
# Prends trop de temps à charger
# plt.subplot(3,3,3)
# sns.histplot(df['BPM'], kde=True, color='#212F45')
plt.subplot(3,3,4)
sns.histplot(df['Anxiety'], kde=True, color='#272640')
plt.subplot(3,3,5)
sns.histplot(df['Depression'], kde=True, color='#312244')
plt.subplot(3,3,6)
sns.histplot(df['Insomnia'], kde=True, color='#3E1F47')
plt.subplot(3,3,7)
sns.histplot(df['OCD'], kde=True, color='#4D194D')
plt.show()