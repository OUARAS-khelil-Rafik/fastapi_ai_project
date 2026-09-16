# Projet complet — Déploiement d'un modèle ML avec FastAPI

## Objectif pédagogique

Construire une API REST capable de prédire le prix d'une maison à partir de :
- surface en m² ;
- nombre de chambres ;
- âge du logement.

Architecture :

Client HTTP → FastAPI → validation Pydantic → pipeline Scikit-learn → prédiction JSON

> Ce projet utilise uniquement FastAPI côté application. Il n'utilise ni Streamlit ni frontend.

---

## 1. Pré-requis

Python 3.10+ recommandé.

Créer et activer un environnement virtuel :

### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows
```powershell
python -m venv .venv
.venv\Scripts\activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
pip install --upgrade pip
```

---

## 2. Entraîner le modèle
et l'exécuter depuis Bash, sans ouvrir Jupyter :

```bash
jupyter nbconvert --to notebook --execute --inplace train.ipynb
```

Résultat attendu :

```text
Modèle entraîné avec succès.
Model saved to: model/model.pkl
R2: ...
MAE: ...
RMSE: ...
```

Le fichier `model/model.pkl` est créé. Il contient la pipeline complète :
prétraitement + modèle de régression.

---

## 3. Démarrer l'API

```bash
uvicorn app.main:app --reload
```

API :

```text
http://127.0.0.1:8000
```

Documentation Swagger :

```text
http://127.0.0.1:8000/docs
```

Documentation OpenAPI alternative :

```text
http://127.0.0.1:8000/redoc
```

---

## 4. Tester l'API

### GET /

```bash
curl http://127.0.0.1:8000/
```

Résultat :

```json
{
  "message": "House Price Prediction API",
  "status": "running"
}
```

### GET /health

```bash
curl http://127.0.0.1:8000/health
```

Résultat :

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST /predict

macOS/Linux :

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"surface":120,"bedrooms":3,"age":7}'
```

Windows PowerShell :

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" `
  -Body '{"surface":120,"bedrooms":3,"age":7}'
```

La réponse contient notamment :

```json
{
  "predicted_price": 250000.0,
  "currency": "DZD",
  "input": {
    "surface": 120.0,
    "bedrooms": 3,
    "age": 7.0
  }
}
```

La valeur exacte dépend du modèle entraîné.

---

## 5. Comprendre le résultat

Le modèle apprend une relation entre les variables d'entrée et le prix.

Entrée :

```text
surface = 120
bedrooms = 3
age = 7
```

Flux :

```text
JSON
 ↓
Pydantic
 ↓
DataFrame
 ↓
Pipeline Scikit-learn
 ↓
LinearRegression
 ↓
prediction
 ↓
JSON
```

`predicted_price` est une estimation, pas un prix réel garanti.

Les métriques du modèle sont affichées après `jupyter nbconvert --to notebook --execute --inplace train.ipynb` :

- MAE : erreur absolue moyenne, dans l'unité du prix ;
- RMSE : pénalise davantage les grandes erreurs ;
- R² : proportion de variance expliquée par le modèle.

---

## 7. Architecture des fichiers

```text
fastapi_ai_project/
├── app/
│   ├── __init__.py            # Initialisation du package
│   └── main.py                # Application FastAPI
├── data/
│   └── houses.csv             # dataset pour entraîner le modèle
├── model/
│   └── model.pkl              # généré par train.ipynb
├── train.ipynb                # notebook pour entraîner le modèle
├── requirements.txt           # dépendances Python
├── .gitignore                 # fichiers à ignorer par Git
└── README.md                  # documentation du projet
```

---

## 8. Point important

L'entraînement et le serveur sont deux étapes différentes :

```text
TRAINING
Dataset → Training → model.pkl

DEPLOYMENT
Client → FastAPI → model.pkl → Prediction
```

On ne réentraîne pas le modèle à chaque requête.