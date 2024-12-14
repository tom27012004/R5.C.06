from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# Charger les données
df = pd.read_csv("../Mental_Health_Music.csv")

# Définir les fonctions manquantes
def calculate_total_variance(data, target):
    return np.var(data[target]) * len(data)

def split_data(data, feature, threshold):
    left = data[data[feature] <= threshold]
    right = data[data[feature] > threshold]
    return left, right

def find_best_split(X, y, min_samples_split):
    """
    Identifier la meilleure division possible pour un nœud.
    
    X : DataFrame des caractéristiques.
    y : Série ou vecteur cible.
    min_samples_split : Nombre minimum d'échantillons requis pour diviser.
    """
    best_feature = None
    best_threshold = None
    best_split = None
    best_loss = float("inf")


    for feature in X.columns:
        # Trier les données selon la caractéristique
        sorted_indices = X[feature].dropna().argsort()
        X_sorted, y_sorted = X.iloc[sorted_indices], y.iloc[sorted_indices]
        
        n_samples = len(X_sorted)
        for i in range(min_samples_split, n_samples - min_samples_split):
            threshold = X_sorted.iloc[i][feature]

            # print(f"Feature: {feature}, Total samples: {len(X_sorted)}, Current index: {i}")

            # Diviser les données
            left_mask = X[feature] <= threshold
            right_mask = X[feature] > threshold

            if sum(left_mask) < min_samples_split or sum(right_mask) < min_samples_split:
                continue

            # Calculer la perte (variance pondérée)
            left_loss = y[left_mask].var() * left_mask.sum()
            right_loss = y[right_mask].var() * right_mask.sum()
            loss = left_loss + right_loss

            # Garder la meilleure division
            if loss < best_loss:
                best_feature = feature
                best_threshold = threshold
                best_loss = loss
                best_split = {"left_mask": left_mask, "right_mask": right_mask}

    return best_feature, best_threshold, best_split


def build_tree(X, y, max_depth=None, min_samples_split=2, min_samples_leaf=1, depth=0):
    """
    Construire un arbre de régression à partir des données.
    
    X : DataFrame des caractéristiques.
    y : Série ou vecteur cible.
    max_depth : Profondeur maximale de l'arbre (None = pas de limite).
    min_samples_split : Nombre minimum d'échantillons pour diviser un nœud.
    min_samples_leaf : Nombre minimum d'échantillons dans une feuille.
    depth : Niveau actuel de l'arbre (interne, utilisé pour contrôler max_depth).
    """
    # Si le nombre d'échantillons est insuffisant ou si on a atteint la profondeur max, retourner une feuille
    if len(y) <= min_samples_leaf or (max_depth is not None and depth >= max_depth):
        return {"value": y.mean()}

    # Trouver la meilleure division
    best_feature, best_threshold, best_split = find_best_split(X, y, min_samples_split)
    
    # Si aucune division valable n'est trouvée, retourner une feuille
    if best_feature is None:
        return {"value": y.mean()}

    # Construire les branches gauche et droite
    left_tree = build_tree(
        X[best_split["left_mask"]],
        y[best_split["left_mask"]],
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        depth=depth + 1,
    )
    right_tree = build_tree(
        X[best_split["right_mask"]],
        y[best_split["right_mask"]],
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        depth=depth + 1,
    )

    # Retourner le nœud de décision
    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "left": left_tree,
        "right": right_tree,
    }


def predict(tree, sample):
    if "value" in tree:
        return tree["value"]
    if sample[tree["feature"]] <= tree["threshold"]:
        return predict(tree["left"], sample)
    else:
        return predict(tree["right"], sample)

def predict_multiple_targets(trees, sample):
    predictions = {}
    for target, tree in trees.items():
        predictions[target] = predict(tree, sample)
    return predictions

# Définir les cibles et les variables explicatives
targets = ["Anxiety", "Depression", "Insomnia", "OCD"]
features = [col for col in df.columns if col not in targets]

# df = df.fillna(df.median())


for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# for target in targets:
#     df[target].hist(bins=30)
#     plt.title(target)
#     plt.show()

# print(df.corr())

# Construire les arbres
trees = {}
max_depth = 4  # Profondeur
min_samples_split = 10  # données nécessaires pour diviser un nœud
min_samples_leaf = 5  # Taille minimale des feuilles
for target in targets:
    print(f"Construction de l'arbre pour {target}")
    trees[target] = build_tree(df[features], df[target], max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)


# Exemple de prédiction
sample = df.iloc[0]  # Exemple de ligne
predictions = predict_multiple_targets(trees, sample)
print("Prédictions pour l'exemple :", predictions)

example_index = 0
real_values = df.iloc[example_index][targets]
print("Valeurs réelles :", real_values)
print("Prédictions :", predictions)

def display_tree(tree, depth=0):
    if "value" in tree:
        print("\t" * depth + f"Value: {tree['value']:.2f}")
        return
    print("\t" * depth + f"Feature: {tree['feature']}, Threshold: {tree['threshold']}")
    print("\t" * depth + "Left:")
    display_tree(tree["left"], depth + 1)
    print("\t" * depth + "Right:")
    display_tree(tree["right"], depth + 1)

# Afficher l'arbre pour une des cibles
# display_tree(trees["Anxiety"])

