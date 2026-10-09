"""Prediction for new cars. The Streamlit app uses this file.

Example:
    from src.predict import predict_price
    result = predict_price(make='Toyota', model_year=2018, mileage=60000, engine_capacity=1300,
                           city='Lahore', fuel_type='Petrol', transmission='Automatic',
                           registered='Lahore', color='White', assembly='Local')
    print(result['price'], result['low'], result['high'])
"""
import json

import joblib
import pandas as pd

from src.config import METADATA_PATH, MODEL_PATH, REF_YEAR
from src.features import FEATURE_COLUMNS, add_derived_features

# what the user has to provide for every car
INPUT_COLUMNS = ['make', 'model_year', 'mileage', 'engine_capacity', 'city',
                 'fuel_type', 'transmission', 'registered', 'color', 'assembly']

_cache = {}


def load_artifacts():
    """Loads the model and its metadata once and keeps them in memory."""
    if 'model' not in _cache:
        _cache['model'] = joblib.load(MODEL_PATH)
        with open(METADATA_PATH) as f:
            _cache['metadata'] = json.load(f)
    return _cache['model'], _cache['metadata']


def predict_dataframe(cars):
    """Predicts a price and a price range for every row of a DataFrame (needs the INPUT_COLUMNS).
    Returns a copy with the new columns: predicted_price, price_low, price_high."""
    missing = [c for c in INPUT_COLUMNS if c not in cars.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')

    model, metadata = load_artifacts()
    ratio_low, ratio_high = metadata['price_range_ratio']

    # build the same features as in training
    cars = cars.reset_index(drop=True)
    X = add_derived_features(cars)[FEATURE_COLUMNS]

    price = model.predict(X)
    result = cars.copy()
    result['predicted_price'] = price.round(-3)
    result['price_low'] = (price * ratio_low).round(-3)
    result['price_high'] = (price * ratio_high).round(-3)
    return result


def predict_price(**car):
    """Predicts one car. Pass the INPUT_COLUMNS as keyword arguments."""
    row = predict_dataframe(pd.DataFrame([car])).iloc[0]
    return {'price': float(row['predicted_price']),
            'low': float(row['price_low']),
            'high': float(row['price_high'])}


def prepare_batch(table):
    """Checks and cleans an uploaded table (for batch prediction).
    Returns the cleaned table and a list of warning messages for the user."""
    missing = [c for c in INPUT_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')

    notes = []
    cars = table[INPUT_COLUMNS].copy()

    for col in ['model_year', 'mileage', 'engine_capacity']:
        cars[col] = pd.to_numeric(cars[col], errors='coerce')

    before = len(cars)
    cars = cars.dropna()
    if len(cars) < before:
        notes.append(f'{before - len(cars)} rows were skipped because of missing or invalid values.')

    # the model was trained on 2024 listings, so newer years are treated as 2024
    too_new = cars['model_year'] > REF_YEAR
    if too_new.any():
        notes.append(f'{too_new.sum()} rows have a model year after {REF_YEAR}; they were treated as {REF_YEAR}.')
        cars.loc[too_new, 'model_year'] = REF_YEAR

    # only "Automatic" is labelled in the training data, everything else is "Unknown"
    cars['transmission'] = cars['transmission'].replace({'Manual': 'Unknown', 'Not Available': 'Unknown'})
    return cars.reset_index(drop=True), notes
