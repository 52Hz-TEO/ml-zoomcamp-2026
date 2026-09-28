import numpy as np
import pandas as pd

df = pd.read_csv("car_fuel_efficiency_2026.csv")

# Q1
version = pd.__version__
print("Q1: " + version)

# Q2
num_records = len(df)
print(f"Q2: {num_records}")

# Q3
print(f"Q3: {df['fuel_type'].nunique()}")

# Q4
cols_with_missing = (df.isnull().sum() > 0).sum()
print(f"Q4: {cols_with_missing}")

# Q5
asia_cars = df[df['origin'] == 'Asia']
max_fuel_eff = asia_cars['fuel_efficiency_mpg'].max()
print('Q5: ' + str(max_fuel_eff))

# Q6
median_hp_before = df["horsepower"].median()
mode_hp = df["horsepower"].mode()[0]

df_filled = df.copy()
df_filled["horsepower"] = df_filled["horsepower"].fillna(mode_hp)
median_hp_after = df_filled["horsepower"].median()

if median_hp_after > median_hp_before:
  change = "Yes, it increased"
elif median_hp_after < median_hp_before:
  change = "Yes, it decreased"
else:
  change = "No"

print(
    f"Q6: {change} (Median before: {median_hp_before}, Median after:"
    f" {median_hp_after})"
)

# Q7
X = asia_cars[["vehicle_weight", "model_year"]].head(7).values

# Normal equation matrix operations: w = (X^T * X)^(-1) * X^T * y
XTX = X.T.dot(X)
XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv.dot(X.T).dot(y)
sum_weights = w.sum()

print(f"Q7: {round(sum_weights, 4)}")