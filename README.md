# R5.C.06
# Analyse des Habitudes d'Écoute Musicale et de la Santé Mentale

## Description
Ce projet explore l'impact des habitudes d'écoute musicale, y compris les genres favoris, sur des indicateurs clés de santé mentale tels que l'anxiété, la dépression, l'insomnie et le TOC (trouble obsessionnel-compulsif). En analysant un jeu de données fourni, nous cherchons à identifier des corrélations significatives entre les comportements d'écoute et le bien-être psychologique.

## Fonctionnalités
- Filtrage des données pour nettoyer les valeurs manquantes.
- Calcul d'un score combiné de santé mentale basé sur l'anxiété, la dépression, l'insomnie et le TOC.
- Analyse et visualisation des données à l'aide de graphiques.
- Étude de corrélations globales et spécifiques à certains genres musicaux (Rap, Classical, Video Game Music).

## Structure des Fichiers
- **`Analyse_global.py`** : Script principal contenant toutes les étapes d'analyse, de la préparation des données à la visualisation des résultats.
- **`Mental_Health_Music.csv`** : Jeu de données source utilisé pour l'analyse.
- **`Arbre_regression.py`** : Script de test avec arbre de régression et tentative de prédiction.

## Prérequis
Avant de lancer le script, assurez-vous d'avoir installé les bibliothèques suivantes :
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`

Vous pouvez installer ces dépendances en exécutant :
```bash
pip install pandas numpy seaborn matplotlib
