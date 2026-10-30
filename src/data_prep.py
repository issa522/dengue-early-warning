"""Data cleaning and feature engineering for the Colombo dengue outbreak early-warning model.

Shared by the notebook and the Streamlit app so that exactly the same
transformations are used for training and for inference.
"""
from datetime import date

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

WEATHER_COLS = ["TEM", "TMAX", "Tm", "SLP", "H", "PP", "VV", "V", "VM"]

# Target: mean weekly cases over the NEXT 4 weeks (t+1 .. t+4).
# Weekly counts are very noisy (reporting delays), a 4-week window matches how
# the Regional Epidemiologist / MOH plans vector-control campaigns.
HORIZON = 4

# Weather variables used as candidate predictors (biologically motivated):
#   PP  rainfall      -> creates breeding sites (log-transformed, very skewed)
#   H   humidity      -> mosquito survival
#   Tm  min temp      -> virus incubation speed inside the mosquito
#   TEM mean temp     -> mosquito development rate
MODEL_WEATHER = ["PP", "H", "Tm", "TEM"]
WEATHER_WINDOWS = [(0, 3), (4, 7)]  # mean over the last 4 weeks, and 4-7 weeks ago

CASE_FEATURES = ["cases_now", "cases_mean2", "cases_mean4", "cases_mean8", "cases_trend"]
SEASON_FEATURES = ["week_sin", "week_cos"]
WEATHER_FEATURES = [f"{c}_lag{lo}_{hi}" for c in MODEL_WEATHER for lo, hi in WEATHER_WINDOWS]

HISTORY_WEEKS = 8  # weeks of history needed to build one feature row


def load_raw(path):
    return pd.read_csv(path)


def clean(raw):
    """Fix the data-quality problems found during EDA.

    1. SLP contains an Excel '#DIV/0!' error -> NaN, then interpolated.
    2. Weekly cases of 0 in 2012 (a record dengue year, neighbouring weeks
       have 60-200 cases) are missing reports, not true zeros -> NaN.
    3. Weeks 37-38 of 2013 are absent -> reindex to a continuous weekly
       calendar.
    4. Weather for 2013-W52 and all of 2014 was imputed by the dataset author
       (almost constant values) -> dropped, we only model observed weather.
    Missing values from 1-3 are filled by linear interpolation in time.
    """
    df = raw.copy()
    df["SLP"] = pd.to_numeric(df["SLP"], errors="coerce")
    df.loc[df["Cases"] == 0, "Cases"] = np.nan

    df["date"] = pd.to_datetime(
        [date.fromisocalendar(int(y), int(w), 1) for y, w in zip(df["Year"], df["Week"])]
    )
    df = df.set_index("date").sort_index()
    df = df[df.index < "2013-12-23"]

    full = pd.date_range(df.index.min(), df.index.max(), freq="W-MON")
    df = df.reindex(full)
    df.index.name = "date"
    iso = df.index.isocalendar()
    df["Year"] = iso.year.astype(int).values
    df["Week"] = iso.week.astype(int).values

    num_cols = ["Cases"] + WEATHER_COLS
    df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")
    return df


def make_features(df, horizon=HORIZON, dropna=True):
    """Turn the weekly series into a supervised table.

    Row t contains only information available at the end of week t, and the
    target is the mean weekly case count over weeks t+1 .. t+horizon.
    """
    out = pd.DataFrame(index=df.index)

    log_cases = np.log1p(df["Cases"])
    out["cases_now"] = log_cases
    out["cases_mean2"] = log_cases.rolling(2).mean()
    out["cases_mean4"] = log_cases.rolling(4).mean()
    out["cases_mean8"] = log_cases.rolling(8).mean()
    out["cases_trend"] = log_cases.rolling(4).mean() - log_cases.shift(4).rolling(4).mean()

    for col in MODEL_WEATHER:
        series = np.log1p(df[col]) if col == "PP" else df[col]
        for lo, hi in WEATHER_WINDOWS:
            out[f"{col}_lag{lo}_{hi}"] = series.shift(lo).rolling(hi - lo + 1).mean()

    # Cyclical encoding of the calendar week (week 52 is next to week 1)
    out["week_sin"] = np.sin(2 * np.pi * df["Week"] / 52)
    out["week_cos"] = np.cos(2 * np.pi * df["Week"] / 52)

    future_mean = df["Cases"][::-1].rolling(horizon).mean()[::-1].shift(-1)
    out["future_mean_cases"] = future_mean
    out["Year"] = df["Year"]
    out["Week"] = df["Week"]
    return out.dropna(subset=CASE_FEATURES + WEATHER_FEATURES) if dropna else out


def features_from_history(cases, weather=None, week=1):
    """Build one feature row from the last 8 weeks of data (used by the app).

    cases   : list of 8 weekly case counts, oldest first
    weather : optional dict {col: list of 8 weekly values, oldest first} for MODEL_WEATHER
              (missing weather is left as NaN and handled by the pipeline's imputer)
    week    : ISO week number of the most recent week
    """
    idx = pd.RangeIndex(HISTORY_WEEKS)
    hist = pd.DataFrame({"Cases": cases, **(weather or {})}, index=idx, dtype=float)
    hist["Year"] = 0  # not used as a feature
    hist["Week"] = [((week - HISTORY_WEEKS + 1 + i - 1) % 52) + 1 for i in range(HISTORY_WEEKS)]
    for col in WEATHER_COLS:
        if col not in hist:
            hist[col] = np.nan
    row = make_features(hist, dropna=False).iloc[[-1]]
    return row[CASE_FEATURES + WEATHER_FEATURES + SEASON_FEATURES]


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Keep only the chosen feature columns (lets one pipeline accept the full feature table)."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Outlier handling: winsorise every column to the [lower, upper] quantiles
    learned on the training data (e.g. extreme rainfall weeks)."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(X, self.lower, axis=0)
        self.high_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return input_features
