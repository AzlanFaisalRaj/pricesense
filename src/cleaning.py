"""Data cleaning. Every step removes rows that are not realistic.
The reasons for each step are explained in the notebook (part 3)."""
import pandas as pd

from src.config import REF_YEAR


def remove_iqr_outliers(data, col):
    """Keeps only rows whose value is inside Q1 - 1.5*IQR ... Q3 + 1.5*IQR."""
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1

    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    kept = data[(data[col] >= low) & (data[col] <= high)]
    print(f'{col}: allowed range {low:,.0f} to {high:,.0f} | removed {len(data) - len(kept)} rows')
    return kept


def clean_data(raw):
    """Raw PakWheels table -> cleaned table. Also returns a log of how many rows each step removed."""
    df = raw.copy()
    log = []

    def record(name, rows_before):
        log.append({'Step': name, 'Rows removed': rows_before - len(df), 'Rows left': len(df)})

    before = len(df)
    df = df.drop_duplicates()
    record('Remove duplicate rows', before)

    before = len(df)
    df = df[df['price'] > 0]                      # price 0 = price was not entered
    record('Remove price = 0', before)

    df = df.rename(columns={'model': 'model_year'})   # the column holds the year, not the model name

    before = len(df)
    df = df[(df['engine_capacity'] >= 600) & (df['engine_capacity'] <= 6000)]
    record('Engine capacity outside 600 to 6000 cc', before)

    before = len(df)
    age = REF_YEAR - df['model_year']
    df = df[~((age > 15) & (df['mileage'] < 5000))]
    record('Older than 15 years with under 5,000 km', before)

    before = len(df)
    df = remove_iqr_outliers(df, 'price')
    record('Price outliers (IQR)', before)

    before = len(df)
    df = remove_iqr_outliers(df, 'mileage')
    record('Mileage outliers (IQR)', before)

    before = len(df)
    age = REF_YEAR - df['model_year']
    df = df[~((df['mileage'] < 100) & (age >= 3))]
    record('Under 100 km but 3+ years old', before)

    # only "Automatic" is labelled in the data, the rest is "Not Available"
    df['transmission'] = df['transmission'].replace('Not Available', 'Unknown')

    # the year at the end of the title must match model_year, otherwise model_year is not reliable
    before = len(df)
    title_year = df['title'].str.extract(r'(\d{4})\s*$')[0].astype(float)
    df = df[title_year == df['model_year']]
    record('Title year does not match model_year', before)

    before = len(df)
    df = df[~((df['price'] < 500000) & (df['model_year'] >= 2012))]
    record('Cars from 2012+ priced under 500,000', before)

    return df.reset_index(drop=True), pd.DataFrame(log)
