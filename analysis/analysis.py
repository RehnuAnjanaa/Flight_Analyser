#!/usr/bin/env python3
"""
analysis.py
Generate static plots and run a RandomForest to identify major factors affecting flight price.

Usage:
  - Place this script in your repo (e.g., analysis/analysis.py).
  - Install requirements: pip install -r requirements.txt
  - Run: python analysis/analysis.py --input flight_pricing_dataset.csv
    or let it download from the repo default RAW_URL.
Outputs:
  - PNG files saved to ./analysis/plots/
  - A simple CSV summary saved to ./analysis/results/feature_importances.csv
"""
import os
import argparse
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

# Default raw URL (reads from your repository root)
RAW_URL = "https://raw.githubusercontent.com/RehnuAnjanaa/EliteHeaven/main/flight_pricing_dataset.csv"

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data(path: Optional[str] = None) -> pd.DataFrame:
    if path is None:
        path = RAW_URL
    print(f"Loading data from: {path}")
    df = pd.read_csv(path, low_memory=False)
    print("Loaded rows,cols:", df.shape)
    return df

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names for common variants
    df = df.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
    # Common mappings used in many flight price datasets
    col_map = {
        "date_of_journey": "date_of_journey",
        "dep_time": "dep_time",
        "departure_time": "dep_time",
        "arrival_time": "arrival_time",
        "duration": "duration",
        "total_stops": "total_stops",
        "stops": "total_stops",
        "airline": "airline",
        "source": "source",
        "destination": "destination",
        "route": "route",
        "price": "price",
        "booking_date": "booking_date",
        "date_of_booking": "booking_date",
    }
    for k, v in col_map.items():
        if k in df.columns and v != k:
            df = df.rename(columns={k: v})
    return df

def parse_dates_and_features(df: pd.DataFrame) -> pd.DataFrame:
    # Work carefully—only create features that we have data for
    df = df.copy()

    # Price numeric
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        raise ValueError("No 'price' column found in dataset. Rename price column to 'Price' or 'price'.")

    # Date of journey -> weekday, month
    if "date_of_journey" in df.columns:
        try:
            df["date_of_journey"] = pd.to_datetime(df["date_of_journey"], dayfirst=True, errors="coerce")
            df["journey_weekday"] = df["date_of_journey"].dt.day_name()
            df["journey_month"] = df["date_of_journey"].dt.month
            df["journey_day"] = df["date_of_journey"].dt.day
            df["is_weekend"] = df["journey_weekday"].isin(["Saturday", "Sunday"])
        except Exception:
            # if parsing fails, skip
            pass

    # Booking date -> days_to_departure
    if "booking_date" in df.columns and "date_of_journey" in df.columns:
        try:
            df["booking_date"] = pd.to_datetime(df["booking_date"], dayfirst=True, errors="coerce")
            df["days_to_departure"] = (df["date_of_journey"] - df["booking_date"]).dt.days
        except Exception:
            pass

    # Parse departure and arrival times (if present)
    for col in ("dep_time", "departure_time"):
        if col in df.columns:
            df["dep_time_parsed"] = pd.to_datetime(df[col], format="%H:%M", errors="coerce").dt.time
            # hour
            df["dep_hour"] = pd.to_datetime(df[col], format="%H:%M", errors="coerce").dt.hour
            break

    if "arrival_time" in df.columns:
        df["arr_hour"] = pd.to_datetime(df["arrival_time"], format="%H:%M", errors="coerce").dt.hour

    # Duration parsing (e.g., "2h 50m" or "2h")
    if "duration" in df.columns:
        def duration_to_minutes(x):
            if pd.isna(x):
                return np.nan
            s = str(x)
            # replace variants
            s = s.replace(" ", "")
            hours = 0
            mins = 0
            try:
                if "h" in s:
                    hpart = s.split("h")[0]
                    hours = int(hpart)
                    rest = s.split("h")[1]
                    if "m" in rest:
                        mins = int(rest.replace("m", "")) if rest.replace("m", "") != "" else 0
                elif "m" in s:
                    mins = int(s.replace("m", ""))
            except Exception:
                return np.nan
            return hours * 60 + mins
        df["duration_mins"] = df["duration"].apply(duration_to_minutes)

    # Total stops parsing (e.g., "non-stop", "1 stop", "2 stops")
    if "total_stops" in df.columns:
        def stops_to_int(x):
            if pd.isna(x):
                return np.nan
            s = str(x).lower().strip()
            if "non" in s or "non-stop" in s or "nonstop" in s:
                return 0
            try:
                # values like "1 stop"
                digits = [int(t) for t in s.split() if t.isdigit()]
                if digits:
                    return digits[0]
                # fallback: extract leading number
                import re
                m = re.search(r"(\d+)", s)
                if m:
                    return int(m.group(1))
            except Exception:
                return np.nan
            return np.nan
        df["stops"] = df["total_stops"].apply(stops_to_int)

    # Route: if there is no route column, create one from source/destination
    if "route" not in df.columns and {"source", "destination"}.issubset(df.columns):
        df["route"] = df["source"].astype(str) + " -> " + df["destination"].astype(str)

    # Sanity: drop rows with missing price
    df = df[~df["price"].isna()].copy()

    return df

def plot_price_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["price"].clip(lower=0), bins=100, kde=True, ax=ax)
    ax.set_xlim(0, df["price"].quantile(0.99))
    ax.set_title("Price distribution (zoomed to 99th pct)")
    ax.set_xlabel("Price")
    fig.savefig(os.path.join(PLOTS_DIR, "price_distribution.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(np.log1p(df["price"].clip(lower=0)), bins=100, kde=True, ax=ax)
    ax.set_title("Log(1+price) distribution")
    ax.set_xlabel("log(1+price)")
    fig.savefig(os.path.join(PLOTS_DIR, "price_distribution_log.png"), bbox_inches="tight")
    plt.close(fig)

def plot_price_by_airline(df: pd.DataFrame, top_n=12):
    if "airline" not in df.columns:
        return
    top_airlines = df["airline"].value_counts().nlargest(top_n).index
    dfsub = df[df["airline"].isin(top_airlines)]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=dfsub, x="airline", y="price", ax=ax)
    ax.set_ylim(0, df["price"].quantile(0.99))
    ax.set_title(f"Price by Airline (top {top_n})")
    plt.xticks(rotation=45)
    fig.savefig(os.path.join(PLOTS_DIR, "price_by_airline_top.png"), bbox_inches="tight")
    plt.close(fig)

def plot_price_by_stops(df: pd.DataFrame):
    if "stops" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x="stops", y="price", ax=ax)
    ax.set_ylim(0, df["price"].quantile(0.99))
    ax.set_title("Price by number of stops")
    fig.savefig(os.path.join(PLOTS_DIR, "price_by_stops.png"), bbox_inches="tight")
    plt.close(fig)

def plot_price_by_weekday(df: pd.DataFrame):
    if "journey_weekday" not in df.columns:
        return
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = plt.subplots(figsize=(9, 5))
    med = df.groupby("journey_weekday")["price"].median().reindex(order)
    med.plot(kind="bar", ax=ax)
    ax.set_title("Median price by weekday of journey")
    ax.set_ylabel("median price")
    fig.savefig(os.path.join(PLOTS_DIR, "median_price_by_weekday.png"), bbox_inches="tight")
    plt.close(fig)

def plot_price_vs_days_to_departure(df: pd.DataFrame):
    if "days_to_departure" not in df.columns:
        return
    dfsub = df[df["days_to_departure"].notna()].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=dfsub.sample(frac=min(1, 10000/len(dfsub)), random_state=1), x="days_to_departure", y="price", alpha=0.2, ax=ax)
    med = dfsub.groupby("days_to_departure")["price"].median().reset_index()
    sns.lineplot(data=med, x="days_to_departure", y="price", color="red", ax=ax)
    ax.set_ylim(0, df["price"].quantile(0.99))
    ax.set_title("Price vs days to departure (median in red)")
    fig.savefig(os.path.join(PLOTS_DIR, "price_vs_days_to_departure.png"), bbox_inches="tight")
    plt.close(fig)

def plot_top_routes_price(df: pd.DataFrame, top_n=20):
    if "route" not in df.columns:
        return
    med_route = df.groupby("route")["price"].median().sort_values()
    top = med_route.tail(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    top.plot(kind="barh", ax=ax)
    ax.set_title(f"Top {top_n} most expensive routes by median price")
    fig.savefig(os.path.join(PLOTS_DIR, "top_routes_by_median_price.png"), bbox_inches="tight")
    plt.close(fig)

def plot_correlation(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return
    corr = numeric.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Spearman correlation (numeric features)")
    fig.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), bbox_inches="tight")
    plt.close(fig)

def generate_all_plots(df: pd.DataFrame):
    print("Generating plots to:", PLOTS_DIR)
    plot_price_distribution(df)
    plot_price_by_airline(df)
    plot_price_by_stops(df)
    plot_price_by_weekday(df)
    plot_price_vs_days_to_departure(df)
    plot_top_routes_price(df)
    plot_correlation(df)
    print("Plots generated.")

def prepare_modeling_data(df: pd.DataFrame, max_cat_top=12):
    """
    Returns X, y and list of categorical and numeric columns used for modeling.
    - For categorical columns we keep top max_cat_top categories and group others under 'Other'.
    """
    df = df.copy()
    y = df["price"].values

    # Candidate categorical features (common ones)
    cat_candidates = [c for c in ["airline", "source", "destination", "route", "journey_weekday"] if c in df.columns]
    num_candidates = [c for c in ["duration_mins", "dep_hour", "arr_hour", "days_to_departure", "stops"] if c in df.columns]

    # Create X with these columns
    X = pd.DataFrame()
    for c in num_candidates:
        X[c] = df[c].astype(float)

    for c in cat_candidates:
        ser = df[c].astype(str).fillna("NA")
        top = ser.value_counts().index[:max_cat_top]
        X[c] = ser.where(ser.isin(top), other="Other")

    return X, y, cat_candidates, num_candidates

def train_feature_importance(X: pd.DataFrame, y: np.ndarray, cat_cols, num_cols):
    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="NA")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ], remainder="drop")

    model = Pipeline(steps=[
        ("pre", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training RandomForest...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("R2:", r2_score(y_test, preds), "RMSE:", mean_squared_error(y_test, preds, squared=False))

    # Permutation importance (gives feature importances on original columns after preprocessing)
    print("Computing permutation importances...")
    r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring="neg_mean_squared_error")

    # build feature names
    # numeric names first, then onehot feature names
    transformed_feature_names = []
    if hasattr(model.named_steps["pre"], "get_feature_names_out"):
        # column transformer get_feature_names_out (sklearn >=1.0)
        try:
            transformed_feature_names = model.named_steps["pre"].get_feature_names_out()
        except Exception:
            transformed_feature_names = None

    if transformed_feature_names is None or len(transformed_feature_names) == 0:
        # Fallback: construct names manually
        ohe = model.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
        ohe_cats = []
        try:
            for cats in ohe.categories_:
                ohe_cats.extend([f"{cat}" for cat in cats])
        except Exception:
            ohe_cats = []
        transformed_feature_names = list(num_cols) + ohe_cats

    importances = pd.Series(r.importances_mean, index=transformed_feature_names).sort_values(ascending=False)
    imp_df = importances.reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_df.to_csv(os.path.join(RESULTS_DIR, "feature_importances.csv"), index=False)

    # Plot top features
    topk = imp_df.head(25)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=topk, y="feature", x="importance", ax=ax)
    ax.set_title("Top features by permutation importance (RandomForest)")
    fig.savefig(os.path.join(PLOTS_DIR, "feature_importances.png"), bbox_inches="tight")
    plt.close(fig)

    return model, imp_df

def main(args):
    df = load_data(args.input if args.input else None)
    df = canonicalize_columns(df)
    df = parse_dates_and_features(df)
    # Basic overview
    print("Columns:", df.columns.tolist())
    print("Sample:")
    print(df.head(3).T)

    # Generate plots
    generate_all_plots(df)

    # Modeling + feature importances
    X, y, cat_cols, num_cols = prepare_modeling_data(df)
    if X.shape[1] > 0:
        model, imp_df = train_feature_importance(X, y, cat_cols, num_cols)
        print("Top importances:\n", imp_df.head(15))
    else:
        print("Not enough features for modeling.")

    print("All done. Plots in:", PLOTS_DIR, "Results in:", RESULTS_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Path or URL to CSV file. Defaults to dataset in the repository.")
    args = parser.parse_args()
    main(args)
