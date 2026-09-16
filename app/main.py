"""
ÉTAPE 09 : APPLICATION FASTAPI

Architecture :

Client HTTP
    ↓
FastAPI
    ↓
Pydantic : validation
    ↓
Pandas DataFrame
    ↓
Pipeline Scikit-learn
    ↓
Prediction
    ↓
Réponse JSON

IMPORTANT :
Le modèle est chargé au démarrage du module, puis réutilisé.
Il n'est PAS entraîné à chaque requête.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# ÉTAPE 10 : Localiser le fichier du modèle.
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"


# ---------------------------------------------------------
# ÉTAPE 11 : Charger le modèle.
#
# Si model.pkl n'existe pas, exécutez :
#     jupyter nbconvert --to notebook --execute --inplace train.ipynb
# ---------------------------------------------------------
model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# ÉTAPE 12 : Créer l'application FastAPI.
# Les métadonnées apparaissent dans Swagger.
# ---------------------------------------------------------
app = FastAPI(
    title="House Price Prediction API",
    description=(
        "API pédagogique de prédiction du prix d'une maison avec FastAPI et Scikit-learn."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# ÉTAPE 13 : Définir le schéma d'entrée avec Pydantic.
#
# Le client doit envoyer exactement les informations nécessaires.
# Les contraintes gt/ge permettent à FastAPI de valider automatiquement
# les valeurs avant d'appeler le modèle.
# ---------------------------------------------------------
class HouseInput(BaseModel):
    surface: float = Field(
        ...,
        gt=0, # La surface doit être strictement positive.
        description="Surface du logement en m².", # Description pour Swagger.
        examples=[120], # Exemple pour Swagger.
    )
    bedrooms: int = Field(
        ...,
        gt=0, # Le nombre de chambres doit être strictement positif.
        description="Nombre de chambres.", # Description pour Swagger.
        examples=[3], # Exemple pour Swagger.
    )
    age: float = Field(
        ...,
        ge=0, # L'âge doit être un nombre positif.
        description="Âge du logement en années.", # Description pour Swagger.
        examples=[7], # Exemple pour Swagger.
    )


# ---------------------------------------------------------
# ÉTAPE 14 : Définir le schéma de sortie.
# Cela documente aussi la réponse de l'API dans Swagger.
# ---------------------------------------------------------
class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str
    input: HouseInput


# ---------------------------------------------------------
# ÉTAPE 15 : Endpoint racine.
#
# GET / est pratique pour vérifier rapidement que le serveur répond.
# ---------------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "House Price Prediction API",
        "status": "running",
    }


# ---------------------------------------------------------
# ÉTAPE 16 : Health check.
#
# Un système externe peut appeler /health pour vérifier que l'API
# et son modèle sont disponibles.
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


# ---------------------------------------------------------
# ÉTAPE 17 : Endpoint de prédiction.
#
# Méthode POST :
# Nous envoyons des données au serveur, donc POST est adapté.
#
# Exemple de body :
# {
#   "surface": 120,
#   "bedrooms": 3,
#   "age": 7
# }
# ---------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(data: HouseInput):
    # ÉTAPE 18 : Transformer l'objet Pydantic en DataFrame.
    # Les noms de colonnes doivent correspondre à ceux utilisés pendant
    # l'entraînement.
    input_df = pd.DataFrame(
        [
            {
                "surface": data.surface,
                "bedrooms": data.bedrooms,
                "age": data.age,
            }
        ]
    )

    # ÉTAPE 19 : Demander une prédiction à la pipeline.
    # La pipeline applique automatiquement le preprocessing sauvegardé,
    # puis appelle le modèle.
    prediction = model.predict(input_df)[0]

    # ÉTAPE 20 : Retourner une réponse JSON.
    # float(...) garantit un type JSON simple.
    return {
        "predicted_price": float(prediction),
        "currency": "DZD",
        "input": data,
    }

# Réaliser par : OUARAS Khelil Rafik