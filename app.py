import datetime
import pandas as pd
import streamlit as st
from forecast_utils import (
    normalize_store_dtypes,
    get_features,
    predict,
    render_shap_plot,
    render_feature_importance,
)

try:
    st.set_option('deprecation.showPyplotGlobalUse', False)
except Exception:
    pass


st.title('Rossmann Daily Store Sales Forecaster')

# Collect forecast inputs from user
forecast_date = st.date_input(
    label='Select Forecast Date',
    value=datetime.date(2015, 8, 1),
    min_value=datetime.date(2015, 8, 1),
)

store_id = st.number_input('Enter Store ID', min_value=1, max_value=1115, value=1)

is_promo = st.checkbox('Will the store run a promotion?', value=False)
promo = 1 if is_promo else 0

holiday_option = st.selectbox(
    'Will there be a State Holiday?',
    ('Public holiday', 'Easter holiday', 'Christmas', 'Working Day'),
    index=3,
)
holiday_code_map = {
    'Public holiday': 'a',
    'Easter holiday': 'b',
    'Christmas': 'c',
    'Working Day': 'nil',
}
state_holiday = holiday_code_map[holiday_option]

run_prediction = st.button('Forecast Sales')
predicted_sales = None
show_interpretability = st.checkbox('Show Model Interpretability Plots', value=False)

if run_prediction:
    request_df = pd.DataFrame(
        {'Date': forecast_date, 'Store': store_id, 'Promo': promo, 'StateHoliday': state_holiday},
        index=[0],
    )

    # Merge request inputs with static store metadata
    store_metadata = pd.read_csv('Data/store.csv')
    feature_row = request_df.merge(store_metadata, on='Store').fillna(0)

    feature_row = normalize_store_dtypes(feature_row)
    feature_row = get_features(feature_row)

    predicted_sales = predict(feature_row)
    st.write('Processed Input Features:')
    st.write(feature_row)
    st.write(f'Predicted Sales for Store {store_id} on {forecast_date}: {round(predicted_sales, 4)} €')

    try:
        if show_interpretability:
            st.write('Generating model explanations...')
            st.subheader('SHAP Force Plot')
            render_shap_plot(feature_row)
            st.subheader('XGBoost Feature Importance (Gain)')
            render_feature_importance()
    except Exception:
        st.write('Prediction explanation plots could not be generated.')
