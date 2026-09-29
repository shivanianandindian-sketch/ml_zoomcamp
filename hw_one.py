import pandas as pd
import numpy as np
from numpy.linalg import inv

# Official pinned 2026 dataset
url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"

df = pd.read_csv(url)
df.head()
# Q1
print("Pandas version:", pd.__version__)

# Q2
print("Records:", len(df))

# Q3
print("Fuel types:", df['fuel_type'].unique())
print("Number of fuel types:", df['fuel_type'].nunique())

# Q4
print("Missing-value columns:", df.isnull().any().sum())

# Q5
asia = df[df['origin'] == 'Asia']
print("Max fuel efficiency:", asia['fuel_efficiency_mpg'].max())

# Q6
median_before = df['horsepower'].median()
mode = df['horsepower'].mode()[0]

df['horsepower'] = df['horsepower'].fillna(mode)

median_after = df['horsepower'].median()

print("Before:", median_before)
print("After:", median_after)
print("Increased:", median_after > median_before)

# Q7
asia = df[df['origin'] == 'Asia']

X = asia[['vehicle_weight', 'model_year']].iloc[:7].to_numpy()

XTX = X.T @ X
XTX_inv = inv(XTX)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = XTX_inv @ X.T @ y

print("Weights:", w)
print("Sum of weights:", w.sum())
