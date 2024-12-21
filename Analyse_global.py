# Recharger le fichier et filtrer les données
file_path = '../Mental_Health_Music.csv'
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données
data = pd.read_csv(file_path)

# Filtrer les données nécessaires
filtered_data = data.dropna(subset=["BPM", "Anxiety", "Depression", "Insomnia", "Fav genre", "Primary streaming service", "Hours per day", "While working", "Instrumentalist", "Foreign languages", "Composer", "Exploratory", "Age"])

# Sélectionner les colonnes essentielles pour l'optimisation
filtered_data = filtered_data[["Fav genre", "BPM", "Anxiety", "Depression", "Insomnia", "OCD", "Primary streaming service", "Hours per day", "While working", "Instrumentalist", "Foreign languages", "Composer", "Exploratory", "Age"]]

filtered_data = filtered_data.replace({"Yes": 1, "No": 0})


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
grouped_data = filtered_data.groupby("Fav genre").agg(
    Mental_Health_Score_Mean=('Mental_Health_Score', 'mean'),
    Number_of_People=('Mental_Health_Score', 'size')
).reset_index()

# Trier les genres par score moyen décroissant
grouped_data = grouped_data.sort_values(by="Mental_Health_Score_Mean", ascending=False)

# Afficher les résultats
print(grouped_data)

plt.figure(figsize = (9, 9))

plt.pie(x = filtered_data["Fav genre"].value_counts().values,
        labels = filtered_data["Fav genre"].value_counts().index.values,
        labeldistance = 1.05,
        autopct = "%1.f%%",
        pctdistance = 0.9)

plt.title("Pourcentage de genre favoris")
plt.show()

age_counts = filtered_data["Age"].value_counts()

# Je retire les moins nombreux pour plus de lisibilité
percentmin = 16
filtered_counts = age_counts[age_counts >= percentmin]
other_count = age_counts[age_counts < percentmin].sum()

plt.figure(figsize = (9, 9))

plt.pie(x = filtered_counts.values,
        labels = filtered_counts.index,
        labeldistance = 1.05,
        autopct = "%1.f%%",
        pctdistance = 0.9)

plt.title("Répartition des ages en pourcentage")
plt.show()

plt.figure(figsize=(10, 8), layout="constrained")
plt.subplot(3,3,4)
sns.histplot(filtered_data['Anxiety'], kde=True, color='#272640')
plt.subplot(3,3,5)
sns.histplot(filtered_data['Depression'], kde=True, color='#312244')
plt.subplot(3,3,6)
sns.histplot(filtered_data['Insomnia'], kde=True, color='#3E1F47')
plt.subplot(3,3,7)
sns.histplot(filtered_data['OCD'], kde=True, color='#4D194D')
plt.subplot(3,3,8)
sns.histplot(filtered_data["Mental_Health_Score"], kde=True, color='#212F45')
plt.show()


total_genre = pd.DataFrame({"genre" : []})

# Supprimer les espaces en début et fin de chaîne
total_genre["genre"] = filtered_data["Fav genre"].str.strip()

# Mettre toutes les chaînes en minuscules pour éviter les doublons liés à la casse
total_genre["genre"] = total_genre["genre"].str.lower()

# Supprimer les doublons
total_genre = total_genre.drop_duplicates(subset=['genre'], keep='first')


# Je vais souvent faire des matrices de corrélation
def matrice_corr(df, label) :
    numeric_data = df.select_dtypes(include=[np.number])

    # Calculer la matrice de corrélation pour un sous-ensemble
    correlation_matrix = numeric_data.corr()

    # Pour visualiser la corrélation avec plus d'éléments
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title(label)
    plt.show()



def hoe_genre_cleaned(row):
    genre_list = total_genre["genre"].tolist()
    row = row.strip()
    row= row.lower()
    result = [1 if genre in row else 0 for genre in genre_list]
    return result

# Pour garder une copie non vectorisé
filtered_data_2 = filtered_data.copy()

filtered_data_2["Fav genre"] = filtered_data_2["Fav genre"].apply(hoe_genre_cleaned) 

# Vérifier que toutes les entrées dans "genre" sont des listes
filtered_data_2["Fav genre"] = filtered_data_2["Fav genre"].apply(lambda x: x if isinstance(x, list) else [])
# Trouver la longueur maximale des listes dans la colonne "genre"
max_genres = max(filtered_data_2["Fav genre"].apply(len))
# Compléter les listes pour qu'elles aient toutes la même longueur
filtered_data_2["Fav genre"] = filtered_data_2["Fav genre"].apply(lambda x: x + [0] * (max_genres - len(x)))
# Convertir les listes en colonnes
genre_columns = pd.DataFrame(filtered_data_2["Fav genre"].tolist(), 
                            columns=[f"genre_{i}" for i in range(max_genres)], 
                            index=filtered_data_2.index)
# Supprimer la colonne d'origine et concaténer les nouvelles colonnes
filtered_data_2 = pd.concat([filtered_data.drop(columns=["Fav genre"]), genre_columns], axis=1)



filtered_data_2 = filtered_data_2.drop(["BPM", "Anxiety", "Depression", "Insomnia", "OCD", "Primary streaming service", "Hours per day", "While working", "Instrumentalist", "Foreign languages", "Composer", "Exploratory", "Age"], axis=1)

label = "Matrice de corrélation entre genre et santé mental global"
matrice_corr(filtered_data_2, label)

label = "Matrice de corrélation global"
matrice_corr(filtered_data, label)

plt.barh(grouped_data["Fav genre"], grouped_data["Mental_Health_Score_Mean"])
plt.title('Score mental moyen par rapport au genre favoris')
plt.ylabel('Fav genre')
plt.xlabel('Mental_Health_Score_Mean')
plt.show()

# Isoler l'analyse pour RAP
df_rap = filtered_data[filtered_data["Fav genre"] == "Rap"]
label = "Matrice de corrélation global opur le genre Rap"
matrice_corr(df_rap, label)

# Isoler l'analyse pour Classical
df_pop = filtered_data[filtered_data["Fav genre"] == "Classical"]
label = "Matrice de corrélation global opur le genre Classical"
matrice_corr(df_pop, label)


# Isoler l'analyse pour Video game music
df_hiphop = filtered_data[filtered_data["Fav genre"] == "Video game music"]
label = "Matrice de corrélation global opur le genre Video game music"
matrice_corr(df_hiphop, label)

