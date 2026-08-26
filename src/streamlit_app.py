import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_utils import evaluate_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELLING_DIR = os.path.join(BASE_DIR, "data", "modelling")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(page_title="Malaysian Condo Price Explorer", page_icon="🏢", layout="wide")

# ---------------------------------------------------------------------------
# Registry of the 6 models this project trains. "data" says which feature set
# each model expects: raw X_train.csv (tree models, scale-invariant) or the
# log1p-scaled X_train_scaled.csv (distance/gradient-based models). KNN's own
# StandardScaler is baked into knn_model.pkl as a sklearn Pipeline (see
# model_knn.ipynb), so it needs no separate scaler file or handling here.
# Only models whose .pkl actually exists in models/ show up in the app.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "Linear Regression": {"file": "linear_model.pkl", "data": "scaled"},
    "Ridge Regression": {"file": "ridge_model.pkl", "data": "scaled"},
    "Random Forest": {"file": "random_forest_model.pkl", "data": "raw"},
    "XGBoost": {"file": "xgboost_model.pkl", "data": "raw"},
    "SVR": {"file": "svr_model.pkl", "data": "scaled"},
    "KNN": {"file": "knn_model.pkl", "data": "scaled"},
}

# Matches data_preprocessing.ipynb Section 3.11.2 exactly - fixed rule, not
# data-derived, so it is safe to mirror as a constant here.
SCALE_COLS = ["Bedroom", "Bathroom", "Property Size", "# of Floors",
              "Total Units", "Parking Lot", "Property Age", "Listed_Facility_Count"]

FACILITY_COLS = ["Barbeque_Area", "Club_House", "Gymnasium", "Jogging_Track", "Lift",
                  "Minimart", "Multipurpose_Hall", "Parking", "Playground", "Sauna",
                  "Security", "Squash_Court", "Swimming_Pool", "Tennis_Court"]
NEARBY_COLS = ["Bus_Stop", "Mall", "Park", "School", "Hospital", "Highway", "Railway_Station"]

# State_Johor / PropertyType_Apartment are the drop_first=True baseline
# categories in data_preprocessing.ipynb 3.9.5/3.9.6 - they map to all-zero
# one-hot rows, not their own column.
STATE_OPTIONS = ["Johor", "Kuala Lumpur", "Melaka", "Negeri Sembilan", "Pahang", "Penang",
                  "Perak", "Putrajaya", "Sabah", "Sarawak", "Selangor", "Other", "Unknown"]
PROPERTY_TYPE_OPTIONS = ["Apartment", "Condominium", "Flat", "Service Residence"]

VALID_STATES = {"Selangor", "Penang", "Kuala Lumpur", "Johor", "Sabah", "Sarawak",
                 "Perak", "Kedah", "Pahang", "Negeri Sembilan", "Melaka",
                 "Terengganu", "Kelantan", "Perlis", "Putrajaya", "Labuan"}
VALID_STATES_LOWER = {s.lower(): s for s in VALID_STATES}


def extract_state_from_address(address):
    """Same rule as data_preprocessing.ipynb 3.8.1, reused here only to let
    the Explore page group/filter listings by state - not part of any model's
    training pipeline."""
    if pd.isna(address):
        return np.nan
    for segment in reversed([s.strip() for s in str(address).split(",")]):
        if segment.lower() in VALID_STATES_LOWER:
            return VALID_STATES_LOWER[segment.lower()]
    return np.nan


# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_feature_columns():
    return list(pd.read_csv(os.path.join(MODELLING_DIR, "X_train.csv"), nrows=0).columns)


@st.cache_data
def load_train_test():
    X_train = pd.read_csv(os.path.join(MODELLING_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(MODELLING_DIR, "X_test.csv"))
    X_train_scaled = pd.read_csv(os.path.join(MODELLING_DIR, "X_train_scaled.csv"))
    X_test_scaled = pd.read_csv(os.path.join(MODELLING_DIR, "X_test_scaled.csv"))
    y_train = pd.read_csv(os.path.join(MODELLING_DIR, "y_train.csv"))["price"]
    y_test = pd.read_csv(os.path.join(MODELLING_DIR, "y_test.csv"))["price"]
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


@st.cache_data
def load_cleaned_data():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "houses_cleaned.csv"))
    df["State"] = df["Address"].apply(extract_state_from_address).fillna("Unknown")
    return df


@st.cache_resource
def load_available_models():
    available = {}
    for name, cfg in MODEL_REGISTRY.items():
        path = os.path.join(MODEL_DIR, cfg["file"])
        if not os.path.exists(path):
            continue
        available[name] = {"model": joblib.load(path), "data": cfg["data"]}
    return available


# ---------------------------------------------------------------------------
# Feature reconstruction: turns a user's plain-language property description
# into the exact 51-column row every model expects, matching
# data_preprocessing.ipynb's engineered feature set (Sections 3.8-3.11).
# ---------------------------------------------------------------------------
def build_feature_row(feature_columns, *, bedroom, bathroom, size, floors, total_units,
                       parking, completion_year, land_title, tenure, floor_range,
                       state, property_type, facilities, nearby):
    row = {c: 0 for c in feature_columns}
    row["Bedroom"] = bedroom
    row["Bathroom"] = bathroom
    row["Property Size"] = size
    row["# of Floors"] = floors
    row["Total Units"] = total_units
    row["Parking Lot"] = parking
    row["Property Age"] = max(2023 - completion_year, 0)  # REFERENCE_YEAR=2023, per 3.8.2
    row["Listed_Facility_Count"] = len(facilities)

    for f in facilities:
        row[f"Has_{f}"] = 1
    for n in nearby:
        row[f"Has_{n}"] = 1

    row["Is_Non_Bumi_Lot"] = 1 if land_title == "Non Bumi Lot" else 0
    row["Freehold Indicator"] = 1 if tenure == "Freehold" else 0
    row["Floor_Range_Ordinal"] = {"Low": 1, "Medium": 2, "High": 3}.get(floor_range, 2)

    state_col = "State_" + state.replace(" ", "_")
    if state_col in row:
        row[state_col] = 1
    ptype_col = "PropertyType_" + property_type.replace(" ", "_")
    if ptype_col in row:
        row[ptype_col] = 1

    return pd.DataFrame([row], columns=feature_columns)


def prepare_for_model(raw_row, model_info):
    if model_info["data"] == "raw":
        return raw_row
    scaled = raw_row.copy()
    scaled[SCALE_COLS] = np.log1p(scaled[SCALE_COLS])
    return scaled


def predict_price(model_info, raw_row):
    X = prepare_for_model(raw_row, model_info)
    log_pred = model_info["model"].predict(X)[0]
    return float(np.exp(log_pred))


def dataset_for_model(model_info, X_raw, X_scaled):
    return X_raw if model_info["data"] == "raw" else X_scaled


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_predict(feature_columns, available_models, accent):
    st.header("🏠 Predict Condominium Price")
    st.caption("Enter a property's details, pick one or more trained models, and compare their predictions.")

    if not available_models:
        st.warning("No trained models found in `models/`. Run the model notebooks first.")
        return

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Property**")
            bedroom = st.slider("Bedroom", 1, 10, 3)
            bathroom = st.slider("Bathroom", 1, 8, 2)
            size = st.number_input("Property Size (sq ft)", 280, 10000, 1000, step=50)
            completion_year = st.slider("Completion Year", 1985, 2023, 2015)
        with col2:
            st.markdown("**Building**")
            floors = st.slider("# of Floors", 2, 63, 20)
            total_units = st.number_input("Total Units", 1, 8000, 300, step=10)
            parking = st.slider("Parking Lot", 1, 10, 1)
            state = st.selectbox("State", STATE_OPTIONS, index=STATE_OPTIONS.index("Selangor"))
        with col3:
            st.markdown("**Terms**")
            property_type = st.selectbox("Property Type", PROPERTY_TYPE_OPTIONS)
            tenure = st.radio("Tenure Type", ["Freehold", "Leasehold"], horizontal=True)
            land_title = st.radio("Land Title", ["Non Bumi Lot", "Bumi Lot"], horizontal=True)
            floor_range = st.radio("Floor Range", ["Low", "Medium", "High"], horizontal=True, index=1)

        st.markdown("**Nearby Amenities**")
        nearby_cols = st.columns(len(NEARBY_COLS))
        nearby = [n for n, c in zip(NEARBY_COLS, nearby_cols)
                  if c.checkbox(n.replace("_", " "), key=f"nearby_{n}")]

        st.markdown("**Facilities**")
        fac_cols = st.columns(4)
        facilities = [f for i, f in enumerate(FACILITY_COLS)
                      if fac_cols[i % 4].checkbox(f.replace("_", " "), key=f"fac_{f}")]

        model_choices = st.multiselect("Compare models", list(available_models.keys()),
                                        default=list(available_models.keys()))
        submitted = st.form_submit_button("Predict Price", type="primary", use_container_width=True)

    if not submitted:
        return
    if not model_choices:
        st.error("Select at least one model.")
        return

    raw_row = build_feature_row(
        feature_columns, bedroom=bedroom, bathroom=bathroom, size=size, floors=floors,
        total_units=total_units, parking=parking, completion_year=completion_year,
        land_title=land_title, tenure=tenure, floor_range=floor_range, state=state,
        property_type=property_type, facilities=facilities, nearby=nearby,
    )

    results = {name: predict_price(available_models[name], raw_row) for name in model_choices}

    st.subheader("Predicted Price")
    cols = st.columns(len(results))
    for c, (name, price) in zip(cols, results.items()):
        c.metric(name, f"RM {price:,.0f}")

    if len(results) > 1:
        names, prices = list(results.keys()), list(results.values())
        order = np.argsort(prices)
        names = [names[i] for i in order]
        prices = [prices[i] for i in order]
        fig, ax = plt.subplots(figsize=(8, 0.6 * len(names) + 1))
        colors = sns.light_palette(accent, n_colors=len(names) + 2)[2:]
        ax.barh(names, prices, color=colors)
        for i, v in enumerate(prices):
            ax.text(v, i, f" RM {v:,.0f}", va="center")
        ax.set_xlabel("Predicted Price (RM)")
        ax.set_title("Predicted Price by Model")
        st.pyplot(fig)
        spread = max(prices) - min(prices)
        st.caption(f"Spread across selected models: RM {spread:,.0f} "
                   f"({spread / np.mean(prices) * 100:.1f}% of the average prediction).")


def page_explore(df, accent):
    st.header("📊 Explore the Dataset")
    st.caption(f"{len(df):,} cleaned listings, from `data/processed/houses_cleaned.csv`.")

    with st.sidebar:
        st.subheader("Filters")
        ptypes = sorted(df["Property Type"].dropna().unique())
        ptype_sel = st.multiselect("Property Type", ptypes, default=ptypes)
        price_lo, price_hi = int(df.price.min()), int(df.price.max())
        price_range = st.slider("Price (RM)", price_lo, price_hi, (price_lo, price_hi))
        bed_lo, bed_hi = int(df.Bedroom.min()), int(df.Bedroom.max())
        bed_range = st.slider("Bedroom", bed_lo, bed_hi, (bed_lo, bed_hi))
        tenures = df["Tenure Type"].dropna().unique().tolist()
        tenure_sel = st.multiselect("Tenure Type", tenures, default=tenures)

    filtered = df[
        df["Property Type"].isin(ptype_sel)
        & df["price"].between(*price_range)
        & df["Bedroom"].between(*bed_range)
        & df["Tenure Type"].isin(tenure_sel)
    ]
    st.write(f"**{len(filtered):,}** listings match your filters")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median Price", f"RM {filtered.price.median():,.0f}" if len(filtered) else "-")
    c2.metric("Avg Size", f"{filtered['Property Size'].mean():,.0f} sq ft" if len(filtered) else "-")
    c3.metric("Avg Bedroom", f"{filtered.Bedroom.mean():.1f}" if len(filtered) else "-")
    c4.metric("Freehold %", f"{(filtered['Tenure Type'] == 'Freehold').mean() * 100:.0f}%" if len(filtered) else "-")

    if filtered.empty:
        st.info("No listings match the current filters.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Price Distribution", "Size vs Price", "By State", "Correlation"])
    with tab1:
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(filtered["price"], bins=40, color=accent, ax=ax)
        ax.set_xlabel("Price (RM)")
        ax.set_title("Price Distribution")
        st.pyplot(fig)
    with tab2:
        sample = filtered.sample(min(2000, len(filtered)), random_state=42)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.scatterplot(data=sample, x="Property Size", y="price", hue="Property Type", alpha=0.5, ax=ax)
        ax.set_title("Property Size vs Price")
        st.pyplot(fig)
    with tab3:
        state_summary = (filtered.groupby("State")["price"].agg(["median", "count"])
                          .sort_values("median", ascending=True))
        state_summary = state_summary[state_summary["count"] >= 3]
        fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(state_summary))))
        colors = sns.light_palette(accent, n_colors=len(state_summary) + 2)[2:]
        bars = ax.barh(state_summary.index, state_summary["median"], color=colors)
        ax.bar_label(bars, labels=[f"RM {v:,.0f} (n={n})" for v, n in
                                    zip(state_summary["median"], state_summary["count"])], padding=3, fontsize=8)
        ax.set_xlabel("Median Price (RM)")
        ax.set_title("Median Price by State (states with >=3 listings)")
        st.pyplot(fig)
    with tab4:
        numeric_cols = ["price", "Property Size", "Bedroom", "Bathroom", "Total Units", "# of Floors", "Parking Lot"]
        corr = filtered[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)

    with st.expander("View filtered listings"):
        st.dataframe(filtered.drop(columns=["description"], errors="ignore"), use_container_width=True)


def page_performance(available_models, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, accent):
    st.header("📈 Model Performance")
    st.caption("Train/Test metrics are computed live from the saved models, so this stays in sync as teammates "
               "retrain and re-save their `.pkl` files.")

    if not available_models:
        st.warning("No trained models found in `models/`.")
        return

    all_names = list(MODEL_REGISTRY.keys())
    missing = [n for n in all_names if n not in available_models]
    if missing:
        st.info(f"Not yet trained/saved: {', '.join(missing)}. They'll appear here automatically once "
                f"their `.pkl` is added to `models/`.")

    rows = []
    for name, info in available_models.items():
        Xtr = dataset_for_model(info, X_train, X_train_scaled)
        Xte = dataset_for_model(info, X_test, X_test_scaled)
        train_m = evaluate_model(info["model"], Xtr, y_train)
        test_m = evaluate_model(info["model"], Xte, y_test)
        rows.append({
            "Model": name,
            "Train RMSE": train_m["RMSE"], "Train R2": train_m["R2"],
            "Test RMSE": test_m["RMSE"], "Test MAE": test_m["MAE"],
            "Test MAPE": test_m["MAPE"], "Test R2": test_m["R2"],
        })
    metrics_df = pd.DataFrame(rows).sort_values("Test RMSE")

    st.subheader("Metrics Table")
    st.dataframe(
        metrics_df.style.format({
            "Train RMSE": "RM {:,.0f}", "Test RMSE": "RM {:,.0f}", "Test MAE": "RM {:,.0f}",
            "Train R2": "{:.3f}", "Test R2": "{:.3f}", "Test MAPE": "{:.1f}%",
        }),
        use_container_width=True, hide_index=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 0.6 * len(metrics_df) + 1))
        ordered = metrics_df.sort_values("Test RMSE", ascending=True)
        colors = sns.light_palette(accent, n_colors=len(ordered) + 2)[2:]
        bars = ax.barh(ordered["Model"], ordered["Test RMSE"], color=colors)
        ax.bar_label(bars, labels=[f"RM {v:,.0f}" for v in ordered["Test RMSE"]], padding=3, fontsize=8)
        ax.set_xlabel("Test RMSE (RM)")
        ax.set_title("Test RMSE by Model (lower is better)")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 0.6 * len(metrics_df) + 1))
        ordered = metrics_df.sort_values("Test R2", ascending=True)
        colors = sns.light_palette(accent, n_colors=len(ordered) + 2)[2:]
        bars = ax.barh(ordered["Model"], ordered["Test R2"], color=colors)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in ordered["Test R2"]], padding=3, fontsize=8)
        ax.set_xlabel("Test R2")
        ax.set_title("Test R2 by Model (higher is better)")
        st.pyplot(fig)

    st.subheader("Feature Importance")
    fi_model = st.selectbox("Model", list(available_models.keys()), key="fi_model")
    info = available_models[fi_model]

    if hasattr(info["model"], "feature_importances_"):
        importances = pd.Series(info["model"].feature_importances_, index=X_train.columns) \
            .sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = sns.light_palette(accent, n_colors=17)[2:]
        ax.barh(importances.index[::-1], importances.values[::-1], color=colors)
        ax.set_xlabel("Importance")
        ax.set_title(f"{fi_model}: Top 15 Features (native importance)")
        st.pyplot(fig)
    elif hasattr(info["model"], "coef_"):
        coefs = pd.Series(np.abs(np.ravel(info["model"].coef_)), index=X_train.columns) \
            .sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = sns.light_palette(accent, n_colors=17)[2:]
        ax.barh(coefs.index[::-1], coefs.values[::-1], color=colors)
        ax.set_xlabel("|Coefficient|")
        ax.set_title(f"{fi_model}: Top 15 Features (coefficient magnitude)")
        st.pyplot(fig)
    else:
        st.caption(f"{fi_model} has no native feature importance (distance/kernel-based model). "
                   "Permutation importance shuffles each feature and measures the resulting drop in Test R2 - "
                   "it can take a few seconds to compute.")
        if st.button("Compute permutation importance", key="perm_btn"):
            from sklearn.inspection import permutation_importance
            Xte = dataset_for_model(info, X_test, X_test_scaled)
            with st.spinner(f"Computing permutation importance for {fi_model}..."):
                perm = permutation_importance(info["model"], Xte, y_test, scoring="r2",
                                               n_repeats=10, random_state=42, n_jobs=-1)
            importances = pd.Series(perm.importances_mean, index=Xte.columns) \
                .sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = sns.light_palette(accent, n_colors=17)[2:]
            ax.barh(importances.index[::-1], importances.values[::-1], color=colors)
            ax.set_xlabel("Mean R2 drop when shuffled")
            ax.set_title(f"{fi_model}: Top 15 Features (permutation importance)")
            st.pyplot(fig)


def main():
    st.sidebar.title("🏢 Malaysian Condo Prices")
    page = st.sidebar.radio("Page", ["🏠 Predict Price", "📊 Explore Dataset", "📈 Model Performance"])
    accent = st.sidebar.color_picker("Chart accent color", "#4C72B0")
    st.sidebar.caption("Customises every chart's color on this page.")

    feature_columns = load_feature_columns()
    available_models = load_available_models()
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = load_train_test()
    cleaned_df = load_cleaned_data()

    if page == "🏠 Predict Price":
        page_predict(feature_columns, available_models, accent)
    elif page == "📊 Explore Dataset":
        page_explore(cleaned_df, accent)
    else:
        page_performance(available_models, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, accent)


if __name__ == "__main__":
    main()
