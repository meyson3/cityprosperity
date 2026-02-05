import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

# Raw columns you have in the CSV (keep these if you still want z-scores for them)
features = [
    "median_age",
    "number_of_births",
    "total_deaths",
    "population_private_household",
    "isced_edu",
    "economically_active_people",
    "gdp",
    "population"
]

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std

def normalize_features(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}\nAvailable: {df.columns.tolist()}")

    for c in cols:
        df[c + "_zscore"] = zscore(df[c])
    return df

def birth_to_death_ratio(df: pd.DataFrame) -> pd.Series:
    return df["number_of_births"] / df["total_deaths"].replace(0, np.nan)

def death_per_capita(df: pd.DataFrame) -> pd.Series:
    return df["total_deaths"] / df["population"].replace(0, np.nan) * 100000

def edu_per_capita(df: pd.DataFrame) -> pd.Series:
    return df["isced_edu"] / df["population"].replace(0, np.nan) * 100000

def economically_active_per_capita(df: pd.DataFrame) -> pd.Series:
    return df["economically_active_people"] / df["population"].replace(0, np.nan) * 100000

def gdp_per_capita(df: pd.DataFrame) -> pd.Series:
    return df["gdp"] / df["population"].replace(0, np.nan)

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bdr"] = birth_to_death_ratio(df)
    df["death_per_100k"] = death_per_capita(df)
    df["edu_per_100k"] = edu_per_capita(df)
    df["active_per_100k"] = economically_active_per_capita(df)
    df["gdp_pc"] = gdp_per_capita(df)
    return df

def build_prosperity_index(df: pd.DataFrame,
                           w_gdp_pc=0.45, w_active=0.25, w_edu=0.20, w_pop=0.10) -> pd.DataFrame:
    """
    PI is a weighted sum of z-scores.
    - death_per_100k is "bad" so we flip its z-score sign.
    """
    df = df.copy()

    # Normalize only the features that should drive the index (not every raw column)
    pi_cols = ["gdp_pc", "active_per_100k", "edu_per_100k", "population", "death_per_100k"]
    df = normalize_features(df, pi_cols)

    # Flip "bad" metric so higher is better
    df["death_per_100k_zscore_good"] = -df["death_per_100k_zscore"]

    df["PI"] = (
        w_gdp_pc * df["gdp_pc_zscore"]
        + w_active * df["active_per_100k_zscore"]
        + w_edu * df["edu_per_100k_zscore"]
        + w_pop * df["population_zscore"]
        + 0.0 * df["death_per_100k_zscore_good"]  # keep 0.0 if you don't want it in PI yet
    )
    return df

def add_two_year_growth_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # GDP per capita 2-year growth
    df["gdp_pc_tplus2"] = df["gdp_pc"].shift(-2)
    df["g2_gdp_pc"] = (df["gdp_pc_tplus2"] - df["gdp_pc"]) / df["gdp_pc"]

    # Education per 100k 2-year growth
    df["edu_pc_tplus2"] = df["edu_per_100k"].shift(-2)
    df["g2_edu_pc"] = (df["edu_pc_tplus2"] - df["edu_per_100k"]) / df["edu_per_100k"]

    # Active per 100k 2-year growth
    df["active_pc_tplus2"] = df["active_per_100k"].shift(-2)
    df["g2_active_pc"] = (df["active_pc_tplus2"] - df["active_per_100k"]) / df["active_per_100k"]

    return df

def fit_pi_to_growth(df: pd.DataFrame, target_col: str):
    train = df.dropna(subset=["PI", target_col]).copy()
    X = train[["PI"]].values
    y = train[target_col].values

    model = LinearRegression()
    model.fit(X, y)

    a = float(model.coef_[0])
    b = float(model.intercept_)
    return model, a, b

def main():
    df = pd.read_csv("stuttgart_clean.csv").sort_values("year").reset_index(drop=True)

    # 1) Engineer features (per-capita + bdr + gdp per capita)
    df = add_engineered_features(df)

    # 2) Build Prosperity Index from selected normalized features
    df = build_prosperity_index(df)

    # 3) Create 2-year-ahead growth targets
    df = add_two_year_growth_target(df)

    # 4) Fit PI -> 2-year growth mappings (one model per target)
    model_gdp, a_gdp, b_gdp = fit_pi_to_growth(df, "g2_gdp_pc")
    model_edu, a_edu, b_edu = fit_pi_to_growth(df, "g2_edu_pc")
    model_act, a_act, b_act = fit_pi_to_growth(df, "g2_active_pc")

    # 5) Forecast using latest PI
    latest_year = int(df["year"].iloc[-1])
    latest_PI = df["PI"].iloc[-1]

    pred_gdp = model_gdp.predict(np.array([[latest_PI]]))[0]
    pred_edu = model_edu.predict(np.array([[latest_PI]]))[0]
    pred_act = model_act.predict(np.array([[latest_PI]]))[0]

    print(f"Latest year: {latest_year}, latest PI: {latest_PI:.3f}")
    print(f"Predicted 2-year GDP per capita growth: {pred_gdp*100:.2f}%")
    print(f"Predicted 2-year education per 100k growth: {pred_edu*100:.2f}%")
    print(f"Predicted 2-year active per 100k growth: {pred_act*100:.2f}%")

    # OPTIONAL: one combined prosperity % (choose weights you like)
    w_gdp, w_edu, w_act = 0.50, 0.25, 0.25
    pred_prosperity = w_gdp*pred_gdp + w_edu*pred_edu + w_act*pred_act
    print(f"Predicted 2-year Prosperity (combined): {pred_prosperity*100:.2f}%")

    # 6) Save (overwrite)
    out = "stuttgart_with_PI.csv"
    if os.path.exists(out):
        print(f"Warning: {out} already exists and will be overwritten.")
    df.to_csv(out, index=False)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
