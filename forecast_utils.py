import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import xgboost
import shap
from fastai.tabular.all import add_datepart


with open('Model/xgb-full.pkl', 'rb') as model_file:
    sales_model = pickle.load(model_file)

with open('Model/feature-xgb.pkl', 'rb') as feat_file:
    selected_features = pickle.load(feat_file)


def normalize_store_dtypes(df):
    """Cast numeric store metadata columns to 64-bit integers."""
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].astype(np.int64)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].astype(np.int64)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].astype(np.int64)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].astype(np.int64)
    return df


def get_features(df):
    """Build engineered calendar, competition, and promotion features for inference."""
    df = df.copy()
    df = add_datepart(df, 'Date', drop=True).drop('Elapsed', axis=1)

    # Check if the current month falls within the store's Promo2 interval
    month_name_map = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
        5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
        9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec',
    }
    df['MonthInPromoInt'] = df[['PromoInterval', 'Month']].apply(
        lambda row: 1 if str(month_name_map[row['Month']]) in str(row['PromoInterval']).split(',') else 0,
        axis=1,
    )

    # Months elapsed since nearest competitor opened
    df[['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']] = (
        df[['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']].astype(np.int64)
    )
    df['compOpenMonths'] = (
        12 * (df.Year - df.CompetitionOpenSinceYear)
        + (df.Month - df.CompetitionOpenSinceMonth).astype(np.float64)
    )
    df['compOpenMonths'] = df['compOpenMonths'].apply(lambda x: x if 0 < x < 1000 else 0)

    # Months elapsed since Promo2 began
    df[['Promo2SinceWeek', 'Promo2SinceYear']] = (
        df[['Promo2SinceWeek', 'Promo2SinceYear']].astype(np.int64)
    )
    df['promoOpenMonths'] = (
        12 * (df.Year - df.Promo2SinceYear) + abs(df.Week - df.Promo2SinceWeek) / 4
    )
    df['promoOpenMonths'] = df['promoOpenMonths'].apply(lambda x: x if 0 < x < 1000 else 0)

    def categorize_comp_distance(distance):
        if np.isnan(distance):
            return 0
        if distance <= 3000:
            return 1
        if distance <= 9000:
            return 2
        return 3

    df['DistBin'] = df['CompetitionDistance'].apply(categorize_comp_distance)
    df['PromoInterval'] = df['PromoInterval'].astype('category').cat.codes

    # Encode ordinal categorical columns consistently with training
    ordinal_cols = ['StateHoliday', 'StoreType', 'Assortment']
    ordinal_encoding = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'nil': 0}
    for col in ordinal_cols:
        df[col] = df[col].apply(lambda val: ordinal_encoding[val])

    df['Week'] = df['Week'].astype(np.int64)
    df = df[selected_features]
    return df


def predict(model_input):
    """Return exponentiated daily sales forecast from log-scale model output."""
    log_prediction = sales_model.predict(model_input).item()
    return np.exp(log_prediction)


def render_shap_plot(model_input):
    """Render a SHAP force plot explaining the predicted store sales."""
    shap_explainer = shap.TreeExplainer(sales_model)
    shap_vals = shap_explainer.shap_values(model_input)
    force_fig = shap.force_plot(
        shap_explainer.expected_value,
        shap_vals,
        model_input.iloc[0],
        matplotlib=True,
        figsize=(15, 8),
    )
    st.pyplot(force_fig)
    plt.clf()


def render_feature_importance():
    """Plot XGBoost feature importance ranked by gain."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    xgboost.plot_importance(sales_model, ax=ax, importance_type='gain', show_values=False)
    st.pyplot(fig)
