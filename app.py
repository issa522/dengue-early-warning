"""Streamlit front end for the Colombo dengue outbreak early-warning model.

Run with:  streamlit run app.py
"""
from datetime import date, timedelta
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.data_prep import HISTORY_WEEKS, MODEL_WEATHER, clean, features_from_history, load_raw

ROOT = Path(__file__).parent

st.set_page_config(page_title="Dengue Early Warning – Colombo", page_icon="🦟", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "models" / "dengue_alert_model.joblib")


@st.cache_data
def load_history():
    return clean(load_raw(ROOT / "data" / "raw" / "colombo_dengue_weather.csv"))


bundle = load_bundle()
model = bundle["model"]
selected = bundle["selected_features"]
uses_weather = any(f.split("_lag")[0] in MODEL_WEATHER for f in selected)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About the model")
    st.markdown(
        f"""
- **Model:** tuned {bundle['model_name']} (scikit-learn pipeline)
- **Trained on:** {bundle['trained_on']}
- **Predicts:** will the average weekly cases over the **next {bundle['horizon_weeks']} weeks** exceed
  **{bundle['high_risk_threshold']:.0f} cases/week**?
- **Alert cut-off:** probability ≥ {bundle['decision_cutoff']:.2f}
"""
    )
    st.subheader("Performance on unseen 2013 data")
    m = bundle["test_metrics"]
    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC", f"{m['ROC-AUC']:.2f}")
    c2.metric("F1", f"{m['F1']:.2f}")
    c1.metric("Recall", f"{m['Recall']:.2f}")
    c2.metric("Precision", f"{m['Precision']:.2f}")
    st.caption("Data: Epidemiology Unit, Ministry of Health Sri Lanka (via the colmozzie dataset).")

# ---------------------------------------------------------------- inputs
st.title("🦟 Dengue Outbreak Early Warning – Colombo District")
st.write(
    "Enter the notified dengue cases for the **last 8 weeks**. The model estimates the probability "
    "that the **next 4 weeks** will be a high-dengue period, so vector-control teams can act early."
)

hist = load_history()
examples = {
    "Custom (type your own)": None,
    "Rising – start of 2013 mid-year outbreak (week of 2013-06-03)": "2013-06-03",
    "Quiet – April 2013 (week of 2013-04-15)": "2013-04-15",
    "Peak – July 2011 outbreak (week of 2011-06-27)": "2011-06-27",
}
choice = st.selectbox("Load an example from the historical data, or enter your own:", list(examples))

if examples[choice]:
    end = pd.Timestamp(examples[choice])
    window = hist.loc[:end].tail(HISTORY_WEEKS)
    default_cases = window["Cases"].round().astype(int).tolist()
    default_date = end.date()
    default_weather = {c: window[c].round(1).tolist() for c in MODEL_WEATHER}
else:
    default_cases = [80, 85, 90, 95, 100, 110, 115, 120]
    default_date = date.today()
    default_weather = {"PP": [5.0] * 8, "H": [80.0] * 8, "Tm": [25.0] * 8, "TEM": [27.5] * 8}

last_week_date = st.date_input("Date of the most recent week (any day in that week)", value=default_date)
iso_week = last_week_date.isocalendar()[1]

st.subheader("Weekly notified cases (oldest → most recent)")
cols = st.columns(HISTORY_WEEKS)
cases = []
for i, c in enumerate(cols):
    week_start = last_week_date - timedelta(weeks=HISTORY_WEEKS - 1 - i)
    cases.append(c.number_input(f"W{week_start.isocalendar()[1]}", min_value=0, max_value=5000,
                                value=int(default_cases[i]), step=1, key=f"c{i}{choice}"))

weather = None
if uses_weather:
    with st.expander("Weekly weather (oldest → most recent)", expanded=False):
        weather = {}
        for col in MODEL_WEATHER:
            wcols = st.columns(HISTORY_WEEKS)
            weather[col] = [wcols[i].number_input(f"{col} {i + 1}", value=float(default_weather[col][i]),
                                                  key=f"{col}{i}{choice}") for i in range(HISTORY_WEEKS)]
else:
    st.caption("ℹ️ Weather did not improve accuracy in validation once recent case counts were known, "
               "so the final model uses case history and season only.")

# ---------------------------------------------------------------- prediction
if any(c == 0 for c in cases):
    st.warning("A week with 0 cases is usually a missing report in Colombo, so check the numbers.")

x = features_from_history(cases, weather, week=iso_week)
prob = float(model.predict_proba(x)[0, 1])
alert = prob >= bundle["decision_cutoff"]

st.divider()
left, right = st.columns([1, 2])
with left:
    st.subheader("Prediction")
    st.metric("Probability of a high-risk month", f"{prob:.0%}")
    st.progress(min(prob, 1.0))
    if alert:
        st.error(f"⚠️ **HIGH RISK.** Expect more than ~{bundle['high_risk_threshold']:.0f} cases/week "
                 "over the next 4 weeks. Recommend intensifying vector control and alerting hospitals.")
    else:
        st.success("✅ **Normal risk** expected for the next 4 weeks. Continue routine surveillance.")

with right:
    fig, ax = plt.subplots(figsize=(7, 3))
    weeks = [f"W{(last_week_date - timedelta(weeks=HISTORY_WEEKS - 1 - i)).isocalendar()[1]}"
             for i in range(HISTORY_WEEKS)]
    ax.bar(weeks, cases, color=["#e74c3c" if c > bundle["high_risk_threshold"] else "#3498db" for c in cases])
    ax.axhline(bundle["high_risk_threshold"], ls="--", color="grey", lw=1, label="High-risk threshold")
    ax.set_ylabel("Cases")
    ax.set_title("Last 8 weeks")
    ax.legend()
    st.pyplot(fig)

# ---------------------------------------------------------------- explanation
st.subheader("Why did the model predict this? (SHAP)")
pre, est = model[:-1], model.named_steps["model"]
x_t = pd.DataFrame(pre.transform(x), columns=selected)
explainer = shap.TreeExplainer(est)
sv = explainer(x_t)
if sv.values.ndim == 3:
    sv = sv[:, :, 1]
# The model works on log-scaled / sin-cos features; show the same values in everyday units.
row = x[selected].iloc[0]
readable = {
    "cases_now": ("cases this week", np.expm1(row.get("cases_now", np.nan))),
    "cases_mean2": ("avg cases, last 2 wks", np.expm1(row.get("cases_mean2", np.nan))),
    "cases_mean4": ("avg cases, last 4 wks", np.expm1(row.get("cases_mean4", np.nan))),
    "cases_mean8": ("avg cases, last 8 wks", np.expm1(row.get("cases_mean8", np.nan))),
    "cases_trend": ("% change vs previous 4 wks", 100 * np.expm1(row.get("cases_trend", np.nan))),
    "week_sin": ("season: week of year", iso_week),
    "week_cos": ("season: week of year ", iso_week),  # trailing space keeps the two labels distinct
}
names = [readable[f][0] if f in readable else f for f in selected]
shown = np.array([readable[f][1] if f in readable else row[f] for f in selected], dtype=float).round(0)
exp = shap.Explanation(values=sv.values[0], base_values=float(np.ravel(sv.base_values)[0]),
                       data=shown, feature_names=names)
fig2 = plt.figure()
shap.plots.waterfall(exp, max_display=10, show=False)
st.pyplot(fig2, clear_figure=True)
st.caption("The chart starts from the average risk in the training data (bottom) and adds each feature's "
           "contribution to reach this week's prediction (top). Red bars push the risk up, blue bars push it "
           "down. Averages are geometric means of the weekly counts.")

with st.expander("Model input features"):
    st.dataframe(x[selected].round(3))
