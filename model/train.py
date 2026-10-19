import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor
from joblib import dump

# Cargar datos
data = pd.read_csv('data/kc_house_data.csv')
data = data.drop_duplicates().reset_index(drop=True)

# Feature Engineering
data['anio_venta'] = data['date'].str[:4].astype(int)
data['mes_venta'] = data['date'].str[4:6].astype(int)
data['antiguedad'] = data['anio_venta'] - data['yr_built']
data['renovada'] = (data['yr_renovated'] > 0).astype(int)
data['ratio_vecinos'] = data['sqft_living'] / data['sqft_living15']
data['dist_centro'] = np.sqrt((data['lat'] - 47.6062) ** 2 + (data['long'] + 122.3321) ** 2)
data['sqft_total'] = data['sqft_living'] + data['sqft_lot']
data['bano_por_hab'] = data['bathrooms'] / (data['bedrooms'] + 1)
data['diag_1'] = data['lat'] + data['long']
data['diag_2'] = data['lat'] - data['long']

# Guardar columnas base
cols_base = [c for c in data.columns if c not in ('id', 'date', 'price')]
X = data[cols_base]
y = data.price

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

# Nearest Neighbors
nn = NearestNeighbors(n_neighbors=16)
nn.fit(data.loc[X_train.index, ['lat', 'long']])

def vecinos(fit_idx, apply_idx, quitar_self=False, k=15):
    dist, ind = nn.kneighbors(data.loc[apply_idx, ['lat', 'long']])
    if quitar_self:
        dist, ind = dist[:, 1:], ind[:, 1:]
    else:
        dist, ind = dist[:, :k], ind[:, :k]
    lp = np.log(y.loc[fit_idx]).values[ind]
    ppsf = (np.log(y.loc[fit_idx]) - np.log(data.loc[fit_idx, 'sqft_living'])).values[ind]
    return pd.DataFrame({
        'vec_logprecio_medio': lp.mean(1),
        'vec_logprecio_mediana': np.median(lp, 1),
        'vec_precio_sqft': ppsf.mean(1),
        'vec_distancia': dist.mean(1)
    }, index=apply_idx)

X_train = X_train.join(vecinos(X_train.index, X_train.index, quitar_self=True))
X_valid = X_valid.join(vecinos(X_train.index, X_valid.index))

# Guardar la lista COMPLETA de columnas usadas para entrenar
cols_to_use = list(X_train.columns)

X_fit, X_stop, y_fit, y_stop = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

configs = [
    dict(max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=0),
    dict(max_depth=8, learning_rate=0.03, subsample=0.7, colsample_bytree=0.7, random_state=1),
    dict(max_depth=5, learning_rate=0.04, subsample=0.8, colsample_bytree=0.9, random_state=2),
    dict(max_depth=7, learning_rate=0.03, subsample=0.75, colsample_bytree=0.6, random_state=3),
]

modelos_entrenados = []
preds_log = []

print('Entrenando modelos XGBoost...')
for cfg in configs:
    modelo = XGBRegressor(
        n_estimators=5000,
        min_child_weight=3,
        reg_lambda=1.0,
        n_jobs=4,
        early_stopping_rounds=100,
        **cfg
    )
    modelo.fit(X_fit, np.log(y_fit), eval_set=[(X_stop, np.log(y_stop))], verbose=False)
    modelos_entrenados.append(modelo)
    preds_log.append(modelo.predict(X_valid))

predictions = np.exp(np.mean(preds_log, axis=0))
print('MAE final:', mean_absolute_error(y_valid, predictions))

pathlib.Path('model').mkdir(exist_ok=True)

print('Guardando artefactos en model/kc_house_xgboost.joblib...')
dump({
    'models': modelos_entrenados,
    'nn': nn,
    'cols_to_use': cols_to_use,
    'y_train_log': np.log(y.loc[X_train.index]).values,
    'sqft_living_train': data.loc[X_train.index, 'sqft_living'].values
}, 'model/kc_house_xgboost.joblib')

print('¡Modelo guardado con éxito!')
