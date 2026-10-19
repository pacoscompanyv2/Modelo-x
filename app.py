import pathlib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from fastapi.middleware.cors import CORSMiddleware

artefactos = load(pathlib.Path('model/kc_house_xgboost.joblib'))
models = artefactos['models']
nn = artefactos['nn']
cols_to_use = artefactos['cols_to_use']
y_train_log = artefactos['y_train_log']
sqft_living_train = artefactos['sqft_living_train']

app = FastAPI(title="King County House Price Prediction")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get('/health')
def health():
    return {"status": "ok"}

class HouseInput(BaseModel):
    date: str = "20141013T000000"
    bedrooms: int = 3
    bathrooms: float = 1.0
    sqft_living: int = 1180
    sqft_lot: int = 5650
    floors: float = 1.0
    waterfront: int = 0
    view: int = 0
    condition: int = 3
    grade: int = 7
    sqft_above: int = 1180
    sqft_basement: int = 0
    yr_built: int = 1955
    yr_renovated: int = 0
    zipcode: int = 98178
    lat: float = 47.5112
    long: float = -122.257
    sqft_living15: int = 1340
    sqft_lot15: int = 5650

class PredictionOutput(BaseModel):
    predicted_price: float

@app.post('/predict_price', response_model=PredictionOutput)
def predict(house: HouseInput):
    data_dict = house.dict()
    df = pd.DataFrame([data_dict])
    
    # Feature Engineering
    df['anio_venta'] = df['date'].str[:4].astype(int)
    df['mes_venta'] = df['date'].str[4:6].astype(int)
    df['antiguedad'] = df['anio_venta'] - df['yr_built']
    df['renovada'] = (df['yr_renovated'] > 0).astype(int)
    df['ratio_vecinos'] = df['sqft_living'] / df['sqft_living15']
    df['dist_centro'] = np.sqrt((df['lat'] - 47.6062) ** 2 + (df['long'] + 122.3321) ** 2)
    df['sqft_total'] = df['sqft_living'] + df['sqft_lot']
    df['bano_por_hab'] = df['bathrooms'] / (df['bedrooms'] + 1)
    df['diag_1'] = df['lat'] + df['long']
    df['diag_2'] = df['lat'] - df['long']

    # K-NN Vecinos
    dist, ind = nn.kneighbors(df[['lat', 'long']])
    idx = ind[0][:15]
    dists = dist[0][:15]
    
    lp = y_train_log[idx]
    ppsf = lp - np.log(sqft_living_train[idx])

    df['vec_logprecio_medio'] = float(lp.mean())
    df['vec_logprecio_mediana'] = float(np.median(lp))
    df['vec_precio_sqft'] = float(ppsf.mean())
    df['vec_distancia'] = float(dists.mean())

    # Reordenar las columnas exactamente en la secuencia original
    X_input = df[cols_to_use]

    # Predicción
    preds_log = [float(m.predict(X_input)[0]) for m in models]
    precio_final = float(np.exp(np.mean(preds_log)))

    return {'predicted_price': round(precio_final, 2)}
