import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
from model_utils import evaluate_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "src")
MODELLING_DIR = os.path.join(BASE_DIR, "model", "modelling_data")
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="NilaiCondo | Malaysian Condo Price Predictor", page_icon="\U0001F3D9️", layout="wide")

# Brand palette - deep pine green + warm brass gold, grounded in Malaysian
# residential real-estate branding rather than a generic dashboard look.
PRIMARY = "#163829"
ACCENT = "#C08A34"
BG = "#FAF7F1"
SURFACE_ALT = "#F3EEE2"
MUTED = "#8A8577"

# ---------------------------------------------------------------------------
# Registry of the 6 models this project trains. "data" says which feature set
# each model expects: raw X_train.csv (tree models, scale-invariant) or the
# log1p-scaled X_train_scaled.csv (distance/gradient-based models). KNN's own
# StandardScaler is baked into knn_model.pkl as a sklearn Pipeline (see
# model_knn.ipynb), so it needs no separate scaler file or handling here.
# Only models whose .pkl actually exists in models/ show up in the app.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "XGBoost": {"file": "xgboost_model.pkl", "data": "raw", "note": "Lowest typical error"},
    "Random Forest": {"file": "random_forest_model.pkl", "data": "raw", "note": "Lowest average error (MAE)"},
    "Ridge Regression": {"file": "ridge_model.pkl", "data": "scaled", "note": "Regularised linear baseline"},
    "Linear Regression": {"file": "linear_model.pkl", "data": "scaled", "note": "Baseline model"},
    "SVR": {"file": "svr_model.pkl", "data": "scaled", "note": "Kernel-based regression"},
    "KNN": {"file": "knn_model.pkl", "data": "scaled", "note": "Similarity-based estimate"},
}
BEST_MODEL = "XGBoost"

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
# Look & feel: one CSS injection, applied to every page. Uses Streamlit's
# stable [data-testid] hooks rather than fragile generated class names, so it
# survives minor Streamlit version bumps.
# ---------------------------------------------------------------------------
def inject_theme():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;0,500;0,600;1,400;1,500&family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{ font-family: 'Manrope', sans-serif; }}
    [data-testid="stAppViewContainer"] {{ background: {BG}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{ padding-top: 1.6rem; max-width: 1180px; }}

    h1, h2, h3 {{ font-family: 'Newsreader', serif !important; font-weight: 500 !important; color: #14201A; }}

    /* Sidebar as the brand nav rail */
    [data-testid="stSidebar"] {{ background: {PRIMARY}; }}
    [data-testid="stSidebar"] * {{ color: #E7F2EA !important; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: #FFFFFF !important; }}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ color: #9CC2AC !important; font-size: 12px; letter-spacing: .04em; text-transform: uppercase; }}
    [data-testid="stSidebar"] [role="radiogroup"] label {{ background: rgba(255,255,255,0.05); border-radius: 10px; padding: 6px 10px; margin-bottom: 4px; }}
    [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.15); }}

    /* Card containers (st.container(border=True)) - soft border, generous
    padding, breathing room between cards so the form doesn't feel boxed-in */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: #FFFFFF; border: 1px solid #F0EBDF !important; border-radius: 18px !important;
        padding: 10px 8px; box-shadow: 0 1px 3px rgba(30,36,31,0.035); margin-bottom: 20px;
    }}
    [data-testid="stForm"] [data-testid="stVerticalBlockBorderWrapper"]:last-of-type {{ margin-bottom: 6px; }}

    /* Tabs - understated, not KPI/dashboard-styled */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap: 28px; border-bottom: 1px solid #E9E2D4; }}
    [data-testid="stTabs"] button[role="tab"] {{ font-weight: 700; font-size: 14px; color: {MUTED}; }}
    [data-testid="stTabs"] button[aria-selected="true"] {{ color: {PRIMARY}; }}
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background-color: {ACCENT} !important; }}

    /* Buttons */
    [data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button {{
        background: {ACCENT}; color: #FFFFFF; border: none; border-radius: 12px; font-weight: 800;
        font-size: 16px; padding: 12px 0; box-shadow: 0 8px 22px -8px rgba(192,138,52,0.55);
    }}
    [data-testid="stButton"] button:hover, [data-testid="stFormSubmitButton"] button:hover {{ background: #A87527; color: #FFFFFF; }}

    /* Sliders / inputs accent */
    [data-testid="stSlider"] [role="slider"] {{ background-color: {ACCENT}; }}
    div[data-baseweb="slider"] > div > div {{ background: {ACCENT} !important; }}
    [data-testid="stCheckbox"] svg, [data-testid="stRadio"] svg {{ fill: {ACCENT} !important; }}

    /* Pills / chip selectors (st.pills - amenities, facilities, model choice, toggles) */
    [data-testid="stButtonGroup"] button {{
        border-radius: 999px !important; font-weight: 600 !important; font-size: 13px !important;
    }}
    [data-testid="stButtonGroup"] button[kind="pillsActive"],
    [data-testid="stButtonGroup"] button[aria-pressed="true"] {{
        background: {PRIMARY} !important; color: #FFFFFF !important; border-color: {PRIMARY} !important;
    }}

    /* Section labels */
    .block-container h5 {{ font-family: 'Manrope', sans-serif !important; font-size: 15px !important;
        font-weight: 800 !important; color: #14201A; text-transform: none; margin-bottom: 2px; }}

    /* Metrics */
    [data-testid="stMetric"] {{ background: #FFFFFF; border: 1px solid #ECE6D9; border-radius: 14px; padding: 14px 16px; }}
    [data-testid="stMetricLabel"] {{ color: {MUTED} !important; }}
    [data-testid="stMetricValue"] {{ color: {PRIMARY} !important; font-family: 'Newsreader', serif; }}

    a {{ color: {ACCENT}; }}
    </style>
    """, unsafe_allow_html=True)


def brand_header(current_page):
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:2px;">
      <span style="font-family:'Newsreader',serif; font-style:italic; font-size:26px; color:{PRIMARY};">NilaiCondo</span>
      <span style="font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:{MUTED}; margin-top:4px;">
        Condominium Price Predictor
      </span>
    </div>
    """, unsafe_allow_html=True)


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
        available[name] = {"model": joblib.load(path), "data": cfg["data"], "note": cfg["note"]}
    return available


@st.cache_data
def load_test_metrics(_available_models, X_test, X_test_scaled, y_test):
    """Live Test RMSE/MAE/R2 per model - computed once (cached) rather than
    hardcoded, so the badges shown in the UI stay in sync with whatever
    .pkl is currently on disk."""
    metrics = {}
    for name, info in _available_models.items():
        Xte = X_test if info["data"] == "raw" else X_test_scaled
        metrics[name] = evaluate_model(info["model"], Xte, y_test)
    return metrics


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
def page_predict(feature_columns, available_models, test_metrics):
    brand_header("predict")

    if not available_models:
        st.warning("No trained models found in `models/`. Run the model notebooks first.")
        return

    model_names = [n for n in MODEL_REGISTRY if n in available_models]
    best = BEST_MODEL if BEST_MODEL in test_metrics else model_names[0]

    # ---- Hero: kept deliberately minimal, no stats, no jargon ----
    st.markdown(f"""
    <div style="padding:22px 0 10px;">
      <div style="font-family:'Newsreader',serif; font-weight:500; font-size:38px; line-height:1.18; color:#14201A; max-width:600px;">
        Know what your condo is worth.
      </div>
      <div style="font-size:15.5px; color:#5C5A50; max-width:560px; margin-top:12px; line-height:1.65;">
        Get an estimated property value based on Malaysian residential listings.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    tab_valuation, tab_compare, tab_how = st.tabs(["Valuation", "Compare Estimates", "How It Works"])

    # ---- Compare Estimates: secondary, optional - lives in its own tab so it
    # never competes with the main valuation flow. The pills widget carries a
    # fixed key so the Valuation tab (rendered below) can read its value. ----
    with tab_compare:
        st.markdown("##### Compare prediction estimates")
        st.caption("Optional. NilaiCondo checks your property against six different prediction estimates "
                    "and always leads with the one it recommends - choose here if you'd like to see more.")
        st.pills(
            "Estimates to include", model_names, selection_mode="multi", default=model_names,
            label_visibility="collapsed", key="model_choices_pills",
        )
        st.markdown(
            f"<div style='font-size:12px; color:{MUTED}; margin-top:10px;'>"
            f"Recommended estimate: <strong style='color:{PRIMARY};'>{best}</strong></div>",
            unsafe_allow_html=True,
        )

    # ---- How It Works: plain-language orientation, no ML terms up front ----
    with tab_how:
        st.markdown("##### Three steps to your estimate")
        st.write("")
        c1, c2, c3 = st.columns(3)
        for col, num, title, desc in [
            (c1, "01", "Describe your unit", "Tell us the size, rooms, location, and what's nearby."),
            (c2, "02", "We check it against real listings", "Your details are compared with thousands of Malaysian condo listings."),
            (c3, "03", "Get your estimate", "See your estimated property value in seconds."),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:#FFFFFF; border:1px solid #ECE6D9; border-radius:14px; padding:20px; height:100%;">
                  <div style="font-family:'Newsreader',serif; color:{ACCENT}; font-size:12px; margin-bottom:10px;">{num}</div>
                  <div style="font-weight:800; font-size:14px; margin-bottom:6px; color:#1E241F;">{title}</div>
                  <div style="font-size:12.5px; color:{MUTED}; line-height:1.5;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        st.write("")
        st.caption("Curious about the six prediction estimates behind this? See the "
                   "“How are these estimates calculated?” section after you get your result.")

    # ---- Valuation: the main journey ----
    with tab_valuation:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap; font-size:11.5px; font-weight:700;
                    letter-spacing:.03em; text-transform:uppercase; color:{MUTED}; margin:6px 0 20px;">
          <span style="color:{PRIMARY};">Property Details</span><span>&rarr;</span>
          <span>Location &amp; Type</span><span>&rarr;</span>
          <span>Amenities &amp; Facilities</span><span>&rarr;</span>
          <span>Your Estimate</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("## Tell us about your property")
        st.caption("Provide a few details about the property to get an estimated market value.")
        st.write("")

        with st.form("predict_form"):
            with st.container(border=True):
                st.markdown("##### Property Details")
                c1, c2, c3 = st.columns(3)
                bedroom = c1.slider("Bedrooms", 1, 10, 3)
                bathroom = c2.slider("Bathrooms", 1, 8, 2)
                parking = c3.slider("Parking", 0, 10, 1)
                c4, c5, c6 = st.columns(3)
                floors = c4.slider("Floors in building", 2, 63, 20)
                total_units = c5.number_input("Total units", 1, 8000, 300, step=10)
                completion_year = c6.slider("Completion year", 1985, 2023, 2015)
                size = st.slider("Property size (sq ft)", 280, 5000, 1000, step=10)

            with st.container(border=True):
                st.markdown("##### Location & Property Type")
                c1, c2 = st.columns(2)
                property_type = c1.selectbox("Property type", PROPERTY_TYPE_OPTIONS)
                state = c2.selectbox("State", STATE_OPTIONS, index=STATE_OPTIONS.index("Selangor"))
                c3, c4, c5 = st.columns(3)
                with c3:
                    floor_range = st.pills("Floor range", ["Low", "Medium", "High"], default="Medium")
                with c4:
                    tenure = st.pills("Tenure", ["Freehold", "Leasehold"], default="Freehold")
                with c5:
                    land_title = st.pills("Land title", ["Non Bumi Lot", "Bumi Lot"], default="Non Bumi Lot")
                floor_range = floor_range or "Medium"
                tenure = tenure or "Freehold"
                land_title = land_title or "Non Bumi Lot"

            with st.container(border=True):
                st.markdown("##### Nearby Amenities")
                st.caption("Select anything within easy reach of the property.")
                nearby_labels = st.pills(
                    "Nearby amenities", [n.replace("_", " ") for n in NEARBY_COLS],
                    selection_mode="multi", default=["Mall", "Railway Station"], label_visibility="collapsed",
                ) or []
                nearby = [n for n in NEARBY_COLS if n.replace("_", " ") in nearby_labels]

            with st.container(border=True):
                st.markdown("##### Facilities")
                st.caption("Select what the building offers residents.")
                default_facility_labels = ["Gymnasium", "Parking", "Playground", "Security", "Swimming Pool"]
                facility_labels = st.pills(
                    "Facilities", [f.replace("_", " ") for f in FACILITY_COLS],
                    selection_mode="multi", default=default_facility_labels, label_visibility="collapsed",
                ) or []
                facilities = [f for f in FACILITY_COLS if f.replace("_", " ") in facility_labels]

            st.write("")
            submitted = st.form_submit_button("Estimate my property value  →", width='stretch', type="primary")

        model_choices = st.session_state.get("model_choices_pills") or model_names

        if not submitted:
            st.markdown(f"""
            <div style="border:1.5px dashed #E3DCCC; border-radius:16px; padding:40px; text-align:center; color:{MUTED}; margin-top:16px;">
              Fill in the details above and click <strong>Estimate my property value</strong> to get your result.
            </div>
            """, unsafe_allow_html=True)
            return

        if not model_choices:
            st.error("Select at least one estimate to include, under the Compare Estimates tab.")
            return

        raw_row = build_feature_row(
            feature_columns, bedroom=bedroom, bathroom=bathroom, size=size, floors=floors,
            total_units=total_units, parking=parking, completion_year=completion_year,
            land_title=land_title, tenure=tenure, floor_range=floor_range, state=state,
            property_type=property_type, facilities=facilities, nearby=nearby,
        )

        results = {name: predict_price(available_models[name], raw_row) for name in model_choices}
        headline_model = best if best in results else max(results, key=results.get)
        headline_price = results[headline_model]
        max_price = max(results.values())

        # ---- Your estimated property value: the strongest visual element ----
        st.write("")
        st.markdown(f"""
        <div style="padding:36px 40px; border-radius:20px; background:linear-gradient(135deg, {PRIMARY} 0%, #1F4D3D 100%); color:#FFF; margin-top:8px;">
          <div style="font-size:12px; letter-spacing:.12em; text-transform:uppercase; font-weight:700; color:#C9E8D5; margin-bottom:10px;">Your Estimated Property Value</div>
          <div style="font-family:'Newsreader',serif; font-weight:500; font-size:50px; color:#FFF;">RM {headline_price:,.0f}</div>
          <div style="font-size:13.5px; color:#C9E8D5; margin-top:12px;">Based on the property details you provided.</div>
          <div style="font-size:12px; color:#C9E8D5; margin-top:14px; opacity:.8;">Powered by {headline_model}</div>
        </div>
        """, unsafe_allow_html=True)

        if len(results) > 1:
            st.write("")
            st.markdown("#### Recommended estimate & other predictions")
            st.caption("Every estimate you chose to include, based on the same property details.")

            for name, price in sorted(results.items(), key=lambda kv: kv[1], reverse=True):
                is_best = name == best
                bar_pct = price / max_price * 100
                bar_color = PRIMARY if is_best else ACCENT
                border = f"border-color:{PRIMARY}; background:#F1F6F2;" if is_best else "border-color:#ECE6D9; background:#FFFFFF;"
                rec_tag = (f"<span style='font-size:9px; font-weight:800; letter-spacing:.04em; text-transform:uppercase; "
                           f"color:#9C6E24; background:#F3E4C5; padding:2px 7px; border-radius:999px; margin-left:6px;'>Recommended estimate</span>"
                           if is_best else "")
                st.markdown(f"""
                <div style="display:grid; grid-template-columns:1fr 1.4fr 150px; align-items:center; gap:18px;
                            padding:15px 20px; border-radius:14px; border:1.5px solid; {border} margin-bottom:10px;">
                  <div style="font-size:14px; font-weight:800; color:#1E241F;">{name}{rec_tag}</div>
                  <div style="height:9px; border-radius:6px; background:#F1EDE2;">
                    <div style="height:100%; width:{bar_pct:.1f}%; border-radius:6px; background:{bar_color};"></div></div>
                  <div style="text-align:right; font-size:15.5px; font-weight:800; color:#1E241F;">RM {price:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

        with st.expander("How are these estimates calculated?"):
            st.markdown(
                "The estimates are generated using six regression models (including XGBoost and Random Forest) "
                "trained on Malaysian condominium listing data (3,755 cleaned listings). Each model independently "
                "learns the relationship between a property's characteristics and its listing price, so their "
                "estimates can differ slightly. R² shows how closely a model's estimates matched real prices "
                "in testing (closer to 1 is better); RMSE shows its typical error size in Ringgit."
            )
            detail_rows = []
            for name in model_names:
                m = test_metrics.get(name)
                if m:
                    detail_rows.append({"Model": name, "R² (accuracy)": round(m["R2"], 3),
                                         "Typical error (RMSE)": f"RM {m['RMSE']:,.0f}"})
            if detail_rows:
                st.dataframe(pd.DataFrame(detail_rows), hide_index=True, width='stretch')

        st.caption("These estimates are for reference only and should not be considered a formal property valuation.")


def page_explore(df, accent):
    brand_header("explore")
    st.header("Market Explorer")
    st.caption(f"{len(df):,} cleaned listings, from `data/processed/houses_cleaned.csv`.")

    with st.sidebar:
        st.markdown("---")
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
        st.dataframe(filtered.drop(columns=["description"], errors="ignore"), width='stretch')


def page_performance(available_models, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, accent):
    brand_header("performance")
    st.header("Model Insights")
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
        width='stretch', hide_index=True,
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


NAV_PREDICT = "Get My Estimate"
NAV_EXPLORE = "Market Explorer"
NAV_PERFORMANCE = "Model Insights"


def main():
    inject_theme()

    with st.sidebar:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; padding:6px 0 2px;">
          <span style="font-family:'Newsreader',serif; font-style:italic; font-size:20px; color:#FFF;">NilaiCondo</span>
        </div>
        <div style="font-size:10.5px; letter-spacing:.08em; text-transform:uppercase; color:#9CC2AC; margin-bottom:14px;">
          Malaysian Condo Valuation
        </div>
        """, unsafe_allow_html=True)
        page = st.radio("Page", [NAV_PREDICT, NAV_EXPLORE, NAV_PERFORMANCE], label_visibility="collapsed")
        st.markdown("---")
        st.caption("Market Explorer and Model Insights are for the project team's own analysis "
                    "and are not part of the buyer-facing valuation flow.")

    feature_columns = load_feature_columns()
    available_models = load_available_models()
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = load_train_test()
    cleaned_df = load_cleaned_data()
    test_metrics = load_test_metrics(available_models, X_test, X_test_scaled, y_test) if available_models else {}

    if page == NAV_PREDICT:
        page_predict(feature_columns, available_models, test_metrics)
    elif page == NAV_EXPLORE:
        page_explore(cleaned_df, ACCENT)
    else:
        page_performance(available_models, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, ACCENT)


if __name__ == "__main__":
    main()
