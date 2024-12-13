import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Charger les données
file_path = './Mental_Health_Music.csv'
data = pd.read_csv(file_path)

# Sélectionner les colonnes utiles
features = data.drop(columns=['Timestamp', 'Anxiety'])  # Supprimer la colonne cible et Timestamp
target = data['Anxiety']

# Identifier les colonnes numériques et catégoriques
num_features = features.select_dtypes(include=['float64']).columns
cat_features = features.select_dtypes(include=['object']).columns

# Préparer les étapes de traitement des données
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Imputation des valeurs manquantes par la moyenne
    ('scaler', StandardScaler())  # Normalisation des variables numériques
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputation des valeurs manquantes par la modalité la plus fréquente
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encodage des variables catégoriques en one-hot
])

# Appliquer les transformations sur les colonnes respectives
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42, max_depth=5, min_samples_split=10))
])


# from sklearn.ensemble import RandomForestRegressor

# model = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', RandomForestRegressor(random_state=42))
# ])

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Entraîner le modèle
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")
