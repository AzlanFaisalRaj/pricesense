"""Feature engineering. The same code is used for training and for the app,
so the model always sees features built in exactly the same way."""
import pandas as pd

from src.config import REF_YEAR

# Brands whose name has two words (the first word alone would be wrong)
MAKE_FIX = {'Mercedes': 'Mercedes Benz', 'Range': 'Range Rover', 'Land': 'Land Rover'}

LUXURY_BRANDS = ['Mercedes Benz', 'BMW', 'Audi', 'Lexus', 'Porsche', 'Land Rover',
                 'Range Rover', 'Volvo', 'Cadillac', 'MINI']

MID_BRANDS = ['Toyota', 'Honda', 'Hyundai', 'KIA', 'Nissan', 'Mitsubishi', 'Mazda', 'Subaru',
              'Ford', 'Chevrolet', 'Changan', 'MG', 'Haval', 'Jeep', 'Peugeot', 'Proton',
              'Chery', 'Volkswagen', 'ORA', 'Renault', 'Fiat', 'SsangYong', 'Isuzu', 'BAIC']

# Columns that are removed before modelling (see the notebook for the reason of each one)
DROP_COLUMNS = ['title', 'price_category', 'mileage_category',
                'post_date', 'post_day_of_week', 'vehicle_age', 'model_year']

NUMERIC_COLUMNS = ['mileage', 'engine_capacity', 'car_age', 'mileage_per_year']
CATEGORICAL_COLUMNS = ['city', 'fuel_type', 'transmission', 'registered',
                       'color', 'assembly', 'make', 'brand_tier']
FEATURE_COLUMNS = ['city', 'mileage', 'fuel_type', 'transmission', 'registered', 'color',
                   'assembly', 'engine_capacity', 'make', 'car_age', 'mileage_per_year', 'brand_tier']


def extract_make(titles):
    """Brand = first word of the title ('Suzuki Vitara GLX 2017' -> 'Suzuki')."""
    return titles.str.split().str[0].replace(MAKE_FIX)


def get_tier(make):
    """Economy, Mid or Luxury. Brands that are not listed are Economy."""
    if make in LUXURY_BRANDS:
        return 'Luxury'
    if make in MID_BRANDS:
        return 'Mid'
    return 'Economy'


def add_derived_features(df):
    """Adds car_age, mileage_per_year and brand_tier.
    The DataFrame needs the columns: model_year, mileage and make."""
    df = df.copy()
    df['car_age'] = REF_YEAR - df['model_year']
    # a car from 2024 has age 0, so use at least 1 to avoid dividing by zero
    df['mileage_per_year'] = df['mileage'] / df['car_age'].clip(lower=1)
    df['brand_tier'] = df['make'].apply(get_tier)
    return df


def build_features(df):
    """For training: cleaned data (with a title column) -> final table with price and the 12 features."""
    df = df.copy()
    df['make'] = extract_make(df['title'])
    df = add_derived_features(df)
    return df.drop(columns=DROP_COLUMNS)
