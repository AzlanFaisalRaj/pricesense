"""Trains the price model from the raw data and saves it.

Run from the project root:
    python -m src.train

It creates models/model.joblib, models/model_metadata.json,
models/model_comparison.csv, models/feature_importance.csv and data/processed/cleaned_cars.csv.
"""
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import features
from src.cleaning import clean_data
from src.config import (METADATA_PATH, MODEL_DIR, MODEL_PATH, PROCESSED_DATA,
                        RANDOM_STATE, RAW_DATA, REF_YEAR)


def build_pipeline(model):
    """Preprocessing + model in one Pipeline. The model learns log(price)."""
    num_steps = Pipeline([
        ('fill', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    # categories with fewer than 30 cars are grouped, unseen categories do not crash
    cat_steps = Pipeline([
        ('fill', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                 min_frequency=30, sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ('num', num_steps, features.NUMERIC_COLUMNS),
        ('cat', cat_steps, features.CATEGORICAL_COLUMNS),
    ])
    target_model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([('prep', preprocess), ('model', target_model)])


def get_models():
    return {
        'Baseline (median)': DummyRegressor(strategy='median'),
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=18, min_samples_leaf=3,
                                               max_features=0.5, n_jobs=-1, random_state=RANDOM_STATE),
        'Gradient Boosting': HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1,
                                                           random_state=RANDOM_STATE),
    }


def main():
    # 1. data
    raw = pd.read_csv(RAW_DATA)
    cleaned, log = clean_data(raw)
    print(log.to_string(index=False))
    print(f'\nRows: {len(raw):,} -> {len(cleaned):,}')

    data = features.build_features(cleaned)
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_DATA, index=False)

    X = data[features.FEATURE_COLUMNS]
    y = data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # 2. compare models with 5-fold cross-validation on the training set only
    scoring = {'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'}
    models = get_models()
    rows = []
    for name, model in models.items():
        scores = cross_validate(build_pipeline(model), X_train, y_train, cv=5, scoring=scoring)
        rows.append({'Model': name, 'MAE': -scores['test_mae'].mean(),
                     'RMSE': -scores['test_rmse'].mean(), 'R2': scores['test_r2'].mean()})
        print(f"{name}: MAE {rows[-1]['MAE']:,.0f} | R2 {rows[-1]['R2']:.3f}")
    results = pd.DataFrame(rows).set_index('Model')

    # 3. final model: best by cross-validation, test set is used once
    best_name = results['MAE'].idxmin()
    final_model = build_pipeline(models[best_name])
    final_model.fit(X_train, y_train)

    pred = final_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, pred))
    test_r2 = r2_score(y_test, pred)
    print(f'\nBest model: {best_name}')
    print(f'Test MAE {test_mae:,.0f} | RMSE {test_rmse:,.0f} | R2 {test_r2:.3f}')

    # 4. price range from out-of-fold predictions on the training set
    oof_pred = cross_val_predict(final_model, X_train, y_train, cv=5)
    ratio_low, ratio_high = (y_train / oof_pred).quantile([0.10, 0.90])
    inside = ((y_test >= pred * ratio_low) & (y_test <= pred * ratio_high)).mean()
    print(f'Price range: x{ratio_low:.3f} to x{ratio_high:.3f} ({inside * 100:.1f}% of test cars inside)')

    # 5. feature importance
    sample = X_test.sample(4000, random_state=RANDOM_STATE)
    imp = permutation_importance(final_model, sample, y_test.loc[sample.index], n_repeats=5,
                                 scoring='neg_mean_absolute_error', random_state=RANDOM_STATE)
    importance = pd.DataFrame({'feature': X.columns, 'mae_increase': imp.importances_mean})
    importance = importance.sort_values('mae_increase', ascending=False)

    # 6. save everything the app needs
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH, compress=3)
    results.round({'MAE': 0, 'RMSE': 0, 'R2': 3}).to_csv(MODEL_DIR / 'model_comparison.csv')
    importance.to_csv(MODEL_DIR / 'feature_importance.csv', index=False)

    metadata = {
        'expected_columns': list(X.columns),
        'numeric_columns': features.NUMERIC_COLUMNS,
        'categorical_options': {c: sorted(X_train[c].unique().tolist()) for c in features.CATEGORICAL_COLUMNS},
        'numeric_ranges': {c: [float(X_train[c].min()), float(X_train[c].max())] for c in features.NUMERIC_COLUMNS},
        'reference_year': REF_YEAR,
        'make_fix': features.MAKE_FIX,
        'luxury_brands': features.LUXURY_BRANDS,
        'mid_brands': features.MID_BRANDS,
        'price_range_ratio': [float(ratio_low), float(ratio_high)],
        'model_name': best_name,
        'test_metrics': {'mae': float(test_mae), 'rmse': float(test_rmse), 'r2': float(test_r2)},
        'versions': {'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'\nSaved the model and metadata in {MODEL_DIR}')


if __name__ == '__main__':
    main()
