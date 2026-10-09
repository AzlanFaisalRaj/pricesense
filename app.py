"""PriceSense: used car price predictor (Streamlit app).
Run with:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from src.config import MODEL_DIR, PROCESSED_DATA, REF_YEAR
from src.features import extract_make, get_tier
from src.predict import INPUT_COLUMNS, load_artifacts, predict_dataframe, predict_price, prepare_batch

st.set_page_config(page_title='PriceSense', page_icon='🚗', layout='wide')

TRANSMISSION_LABELS = {'Automatic': 'Automatic', 'Unknown': 'Manual / not specified'}


# ----------------------------------------------------------------- loading
@st.cache_resource
def get_model_info():
    _, metadata = load_artifacts()
    return metadata


@st.cache_data
def get_options():
    """Values for the dropdowns. The most common values come first."""
    metadata = get_model_info()
    options = dict(metadata['categorical_options'])

    if PROCESSED_DATA.exists():
        data = pd.read_csv(PROCESSED_DATA, usecols=['make', 'city', 'color', 'registered'])
        makes = data['make'].value_counts()
        # brands with fewer than 30 cars are unknown to the model, so they are not offered
        options['make'] = makes[makes >= 30].index.tolist()
        options['city'] = data['city'].value_counts().head(40).index.tolist()
        options['color'] = data['color'].value_counts().head(15).index.tolist()
        options['registered'] = data['registered'].value_counts().head(30).index.tolist()
    return options


@st.cache_data
def read_csv_if_exists(name):
    path = MODEL_DIR / name
    return pd.read_csv(path) if path.exists() else None


def pick(options, wanted):
    """Index of the default value in a dropdown list."""
    return options.index(wanted) if wanted in options else 0


def lakh(value):
    return f'{value / 100000:.1f} lakh'


metadata = get_model_info()
options = get_options()
ranges = metadata['numeric_ranges']
oldest_year = int(REF_YEAR - ranges['car_age'][1])
max_km = int(ranges['mileage'][1])
min_cc, max_cc = int(ranges['engine_capacity'][0]), int(ranges['engine_capacity'][1])

# ----------------------------------------------------------------- header
st.title('🚗 PriceSense')
st.write('Estimate the fair market price of a used car in Pakistan from its brand, age, mileage, engine and city.')

tab_predict, tab_batch, tab_about = st.tabs(['Predict a price', 'Batch prediction (CSV)', 'About the model'])

# ----------------------------------------------------------------- tab 1: single prediction
with tab_predict:
    col1, col2, col3 = st.columns(3)
    with col1:
        make = st.selectbox('Brand', options['make'], index=pick(options['make'], 'Toyota'))
        model_year = st.number_input('Model year', min_value=oldest_year, max_value=REF_YEAR, value=2018, step=1)
        mileage = st.number_input('Mileage (km)', min_value=0, max_value=max_km, value=60000, step=1000)
    with col2:
        engine = st.number_input('Engine capacity (cc)', min_value=min_cc, max_value=max_cc, value=1300, step=100)
        fuel_type = st.selectbox('Fuel type', options['fuel_type'], index=pick(options['fuel_type'], 'Petrol'))
        transmission = st.selectbox('Transmission', list(TRANSMISSION_LABELS),
                                    format_func=lambda x: TRANSMISSION_LABELS[x])
    with col3:
        city = st.selectbox('City', options['city'], index=pick(options['city'], 'Lahore'))
        registered = st.selectbox('Registered in', options['registered'], index=pick(options['registered'], 'Lahore'))
        assembly = st.selectbox('Assembly', options['assembly'], index=pick(options['assembly'], 'Local'))

    color = st.selectbox('Color', options['color'], index=pick(options['color'], 'White'))
    listed_price = st.number_input('Listed price in PKR (optional, to check if it is a good deal)',
                                   min_value=0, value=0, step=50000)

    if st.button('Predict price', type='primary'):
        result = predict_price(make=make, model_year=model_year, mileage=mileage, engine_capacity=engine,
                               city=city, fuel_type=fuel_type, transmission=transmission,
                               registered=registered, color=color, assembly=assembly)

        st.divider()
        m1, m2 = st.columns(2)
        m1.metric('Estimated price', f"PKR {result['price']:,.0f}", lakh(result['price']), delta_color='off')
        m2.metric('Likely range (about 80% of cars)', f"{lakh(result['low'])} to {lakh(result['high'])}")
        st.caption(f"PKR {result['low']:,.0f} to PKR {result['high']:,.0f}")

        if listed_price > 0:
            difference = (listed_price - result['price']) / result['price'] * 100
            if listed_price < result['low']:
                st.success(f'Good deal: the listed price is {abs(difference):.0f}% below the estimate. '
                           'Check the car condition, a very low price can also mean a problem.')
            elif listed_price <= result['high']:
                st.info(f'Fair price: the listed price is inside the likely range ({difference:+.0f}% from the estimate).')
            else:
                st.warning(f'Expensive: the listed price is {difference:.0f}% above the estimate.')

        if get_tier(make) == 'Luxury':
            st.warning('Luxury cars are rare in the training data (about 1%), so treat this estimate with extra caution.')
        st.caption(f'The model was trained on 2024 asking prices, so the estimate is a 2024 price level.')

    st.divider()
    st.subheader('What drives the price?')
    importance = read_csv_if_exists('feature_importance.csv')
    if importance is not None:
        chart = importance.set_index('feature')['mae_increase'] / 100000
        st.bar_chart(chart.sort_values(), horizontal=True)
        st.caption('Permutation importance: how many lakh PKR the average error grows when that column is shuffled. '
                   'A longer bar means the model relies on that column more.')

# ----------------------------------------------------------------- tab 2: batch prediction
with tab_batch:
    st.write('Upload a CSV with one car per row and get a price estimate for each car.')
    st.write('Required columns: ' + ', '.join(f'`{c}`' for c in INPUT_COLUMNS))

    example = pd.DataFrame([
        ['Toyota', 2018, 60000, 1300, 'Lahore', 'Petrol', 'Automatic', 'Lahore', 'White', 'Local'],
        ['Suzuki', 2012, 110000, 1000, 'Karachi', 'Petrol', 'Unknown', 'Karachi', 'Silver', 'Local'],
    ], columns=INPUT_COLUMNS)
    st.download_button('Download an example CSV', example.to_csv(index=False), 'example_cars.csv', 'text/csv')

    uploaded = st.file_uploader('Upload your CSV', type='csv')
    if uploaded is not None:
        try:
            cars, notes = prepare_batch(pd.read_csv(uploaded))
            for note in notes:
                st.warning(note)
            if cars.empty:
                st.error('No valid rows found in the file.')
            else:
                predictions = predict_dataframe(cars)
                st.dataframe(predictions, use_container_width=True)
                st.download_button('Download the predictions', predictions.to_csv(index=False),
                                   'predictions.csv', 'text/csv')
        except ValueError as error:
            st.error(str(error))
        except Exception:
            st.error('Could not read this file. Please check that it is a valid CSV.')

# ----------------------------------------------------------------- tab 3: about
with tab_about:
    metrics = metadata['test_metrics']
    st.subheader('How good is the model?')
    c1, c2, c3 = st.columns(3)
    c1.metric('Test MAE (average error)', lakh(metrics['mae']))
    c2.metric('Test RMSE', lakh(metrics['rmse']))
    c3.metric('Test R2', f"{metrics['r2']:.3f}")
    st.write(f"Final model: **{metadata['model_name']}**. Metrics are measured on cars the model never saw during training.")

    comparison = read_csv_if_exists('model_comparison.csv')
    if comparison is not None:
        st.write('Cross-validation comparison of all models (training data only):')
        st.dataframe(comparison, hide_index=True)

    st.subheader('Limitations')
    st.markdown(
        '- Prices are **2024 asking prices** from PakWheels listings, not final sale prices.\n'
        '- Cars above about 9.4 million PKR were removed as outliers, and only about 1% of the data is luxury. '
        'Estimates for luxury cars are not reliable.\n'
        '- Only the brand is used, not the exact model (Corolla, Civic ...) or the condition of the car.\n'
        '- "Manual / not specified" covers all listings without an "Automatic" label.\n'
        '- The estimate is a guide, not a valuation.')
