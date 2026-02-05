import pandas as pd
import numpy as np

df = pd.read_excel("Stuttgart.xlsx")
df = df.sort_values("year").reset_index(drop=True)

# 1) Fix the obvious outlier (set to NaN so it can be interpolated)
#    (You can also add rules like: if deaths < 200, treat as missing)
df.loc[df["total_deaths"] < 200, "total_deaths"] = np.nan

# 2) Interpolate numeric columns across years
num_cols = [c for c in df.columns if c != "year"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")

print(df.isna().sum())  # should be 0 for most/all columns now
df.to_csv("stuttgart_clean.csv", index=False)