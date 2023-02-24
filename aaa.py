import numpy as np
import pandas as pd

from joblib import dump, load

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import scipy.stats as stats

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

df = pd.read_csv('./data/MotorAlpes_data.csv')

# ----------------------------
# Descripcion de los datos
# ----------------------------

df.shape()
df.info() 
df.describe()

# ----------------------------
# Completitud
# ----------------------------

df.isnull().sum()

# ----------------------------
# Consistencia
# ----------------------------

# check if year is between 1994 and 2020

print(all(map(lambda x: x in range(1994,2021), df["year"])))

# check if km_driven is between 1 and 2’360.457

print(all(map(lambda x: x in range(1,2360458), df["km_driven"])))

# check if owner is in ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car']

print(all(map(lambda x: x in ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], df["owner"])))

# check if seller_type is in ['Individual', 'Dealer', 'Trustmark Dealer']

print(all(map(lambda x: x in ['Individual', 'Dealer', 'Trustmark Dealer'], df["seller_type"])))

# check if seats is between 2 and 14

print(all(map(lambda x: x in range(2,14), df["seats"])))

# check if fuel is in ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']

print(all(map(lambda x: x in ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'], df["fuel"])))

# check if transmission is in ['Manual', 'Automatic']

print(all(map(lambda x: x in ['Manual', 'Automatic'], df["transmission"])))

# check if mileage is between 0 and 46,816

check = True
for i in df["max_power"]:
    if i < 34.2 or i > 400:
        check = False
        break
print(check)

# check if engine is between 624 and 3604

print(all(map(lambda x: x in range(624,3605), df["engine"])))

# check if max_power is between 34.2 and 400

check = True
for i in df["max_power"]:
    if i < 34.2 or i > 400:
        check = False
        break
print(check)

# check if selling_price is between 363.45 and 121153.38

check = True
for i in df["selling_price"]:
    if i < 363.45 or i > 121153.38:
        check = False
        break
print(check)

# ----------------------------
# Consistencia
# ----------------------------

# for every categorical variable, check if there are values that are not in the list of possible values

categorical = ['owner', 'seller_type', 'fuel', 'transmission']

for col in categorical:
    print(col, df[col].unique(dropna = False, normalize = True))


# ----------------------------
# Analisis exploratorio
# ----------------------------

# Preparación de datos

# transform year from float to int

df['year'] = df['year'].astype(int)

# Clean null values

df = df.dropna()

# Data visualization

sns.pairplot(df)
plt.show()





