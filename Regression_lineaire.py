import pandas as pd
import numpy as np

# Charger les données
df = pd.read_csv("../Mental_Health_Music.csv")

# Gérer les valeurs manquantes
df = df.dropna()  # Supprime les lignes avec des valeurs manquantes (ou utilisez une autre stratégie)


# Afficher les colonnes finales
# print(df.columns)

import seaborn as sns
import matplotlib.pyplot as plt

# Calculer la corrélation
# correlation_matrix = df.corr()


# Sélectionner uniquement les variables numériques
numeric_data = df.select_dtypes(include=[np.number])

# Calculer la matrice de corrélation pour un sous-ensemble
correlation_matrix = numeric_data.corr()


# Heatmap pour visualiser la corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Matrice de Corrélation")
plt.show()

# Scatter plot entre une variable explicative et la cible
plt.scatter(df["Age"], df["Anxiety"])
plt.xlabel("Age")
plt.ylabel("Anxiety")
plt.title("Relation entre Variable Explicative et Cible")
plt.show()
