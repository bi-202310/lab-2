# -*- coding: utf-8 -*-
"""regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zk6bYqo2IBW6cGXbqt1nR6-DiMmLGbrQ

# Laboratorio 2

## 1. Carga de datos
##### Se importan todas las librerias necesarias para poder crear, entrenar y crear el modelo
"""

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
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

import scipy.stats as stats

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

df = pd.read_csv('./data/MotorAlpes_data.csv', index_col=False)
df = df.iloc[:, 1:]

"""## 2. Descripción de los datos
##### Se hace una descripcion de los datos para poder entender con que tipo de datos estamos trabajando y elementos que importantes que toque tener en cuenta 
"""

df.shape

df.info()

df.describe()

"""### 2.1 Completitud
##### Revision de datos faltantes y el porcentaje que representan
"""

df.isnull().sum()

# Show null values as a percentage of the dataframe
df.isnull().sum() / df.shape[0]

"""### 2.2 Consistencia
##### Revision de que todos los datos que se encuentran dentro del dataframe cuamplan con las normas establecidas en el diccionario de datos
"""

# check if year is between 1994 and 2020

print(all(map(lambda x: x in range(1994,2021), df["year"])))

# check if km_driven is between 1 and 2’360.457

print(all(map(lambda x: x in range(1,2360458), df["km_driven"])))

# check if owner is in ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car']

print(all(map(lambda x: x in ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], df["owner"])))

# check if seller_type is in ['Individual', 'Dealer', 'Trustmark Dealer']

print(all(map(lambda x: x in ['Individual', 'Dealer', 'Trustmark Dealer'], df["seller_type"])))

# check if seats is between 2 and 14

print(all(map(lambda x: x in range(2,15), df["seats"])))

# check if fuel is in ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']

print(all(map(lambda x: x in ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'], df["fuel"])))

# check if transmission is in ['Manual', 'Automatic']

print(all(map(lambda x: x in ['Manual', 'Automatic'], df["transmission"])))

# check if mileage is between 0 and 46.816

print(all(map(lambda x: 34.2 <= x <= 46.816, df["mileage"])))

# check if engine is between 624 and 3604

print(all(map(lambda x: x in range(624,3605), df["engine"])))

# check if max_power is between 34.2 and 400

print(all(map(lambda x: 32.8 <= x <= 400, df["max_power"])))

# check if selling_price is between 363.45 and 121153.38

print(all(map(lambda x: 363.45 <= x <= 121153.38, df["selling_price"])))

"""### 2.3 Consistencia
##### Revisar que todos los datos sean consistentes semanticamente, es decir, los atributos sean unicos y no tengan otros nombres
"""

# for every categorical variable, check if there are values that are not in the list of possible values

categorical = ['owner', 'seller_type', 'fuel', 'transmission']

for col in categorical:
    print(col, df[col].unique())

"""## 3. Análisis Exploratorio

### 3.1 Preparación de los datos
##### Se deben preparar los datos para poderlos procesar. Esto incluye agregar nuevas columnas relevantes, llenar valores vacios con alguna metrica determinada y reemplazar los valores categoricos por representaciones numericas
"""

# Set a column that specifies the age gap
df['antiquity'] = 2020 - df['year']

# Fill numerical values with the median so as not to skew the data
df = df.fillna(df.median())

# Fill categorical values with the mode
categorical_cols = ['owner', 'seller_type', 'fuel', 'transmission']
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Replace all the categorical values with numbers
df['owner'] = df['owner'].replace({'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5})
df['seller_type'] = df['seller_type'].replace({'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3})
df['fuel'] = df['fuel'].replace({'Petrol': 1, 'Diesel': 2, 'CNG': 3, 'LPG': 4, 'Electric': 5})
df['transmission'] = df['transmission'].replace({'Manual': 1, 'Automatic': 2})

df.info()

# Asumimos que el experto decidio que estos eran los features mas relevantes
features = ["antiquity", "km_driven", "seats", "fuel", "transmission", "max_power"]

# Make a scaler to normalize de data
scaler = StandardScaler()
x = df
y = df['selling_price']
features = ["antiquity", "km_driven", "seats", "fuel", "transmission", "max_power"]
x = scaler.fit_transform(x[features])

# Make a dataframe with the normalized data and keep the old one
og_df = df
df = pd.DataFrame(x, columns=features)
df['selling_price'] = y
df

"""### 3.2 Visualización de los datos
##### Se visualizan los datos incialmente comparados contra el "selling_price" para ver tendencias y ayudar en la identificacion de features
"""

#sns.pairplot(df)
#plt.show()

sns.pairplot(df, x_vars=list(df.columns), y_vars='selling_price', height=4, aspect=1, kind='reg')
sns.pairplot(df, x_vars=list(df.columns), y_vars='selling_price', height=4, aspect=1)

df[df.columns].corr()

# Correlation between features chosen
df[features].corr()

"""## 4. Modelamiento
##### Se procede a realizar el modelo de acuerdo al analisis exploratorio y atributos escogidos como mas relevantes para entrenar el modelo
"""

x_train, x_test, y_train, y_test = train_test_split(df[features], df['selling_price'], test_size = 0.3, random_state = 1)

x_train

x_test

y_train

y_test

x_train.shape, y_train.shape

x_test.shape, y_test.shape

"""### 4.1. Regresion
##### Se procede a realizar una regresion linear para tratar de predecir el "selling_price" de acuerdo con los atributos escogidos
"""

regression = LinearRegression()

regression.fit(x_train, y_train)

regression.intercept_

pd.DataFrame({'columns': features, 'coef': regression.coef_})

# Plot of the regression overlapped with the data in order to see if the regression has some relation with the data
f, axs = plt.subplots(1, len(features[:-1]), sharey = True, figsize = (20, 4))

for i in range(len(features[:-1])):
    col = features[i]
    x = x_train[col]
    m = regression.coef_[i]
    b = regression.intercept_
    axs[i].plot(x, y_train, 'o', alpha = 0.3)
    axs[i].plot(x, x * m + b)
    axs[i].set_title(col)

# Plot only the regression to see clearly how each attribute affects the regression line on a scale that is visibly understandable
f, axs = plt.subplots(1, len(features[:-1]), sharey = True, figsize = (20, 4))

for i in range(len(features[:-1])):
    col = features[i]
    x = x_train[col]
    m = regression.coef_[i]
    b = regression.intercept_
    axs[i].plot(x, x * m + b)
    axs[i].set_title(col)

# Comparing the real values with the predicted values with an difference percentage for each prediction

y_pred = regression.predict(x_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df['Difference'] = df['Actual'] - df['Predicted']
df['Difference %'] = df['Difference'] / df['Actual'] * 100
df

"""### 5. Validación del modelo
##### Se debe revisar que el error del modelo entrenado este cercano al error de los datos de prueba. Esto se hace para verificar que el modelo no tenga overfitting o underfitting ya que esto podria evitar que el modelo prediciera adecuadamente el "selling_price" de datos que nunca haya visto.
"""

print('Mean Absolute Error Train:', mean_absolute_error(y_train, regression.predict(x_train)))
print('Mean Absolute Error Test:', mean_absolute_error(y_test, regression.predict(x_test)))

print('Root Mean Squared Error Train:', np.sqrt(mean_squared_error(y_train, regression.predict(x_train))))
print('Root Mean Squared Error Test:', np.sqrt(mean_squared_error(y_test, regression.predict(x_test))))

"""Podemos apreciar que nuestro error en las pruebas se acerca bastante al error del entrenamiento, y ademas es un error relativamente bajo, lo cual nos da un buen indicio de que nuestro modelo predice adecuadamente el "selling_price" de un carro """

# Plot real value of objective variable on a boxplot

plt.figure(figsize=(20, 5))
sns.boxplot(x = y_test, showmeans = True)
plt.title('Real value of objective variable')
plt.grid()
plt.show()

# Plot |real value - predicted value| of objective variable on a boxplot

plt.figure(figsize=(20, 5))
sns.boxplot(x = abs(y_test - regression.predict(x_test)), showmeans=True)
plt.title('Absolute error of objective variable')
plt.grid()
plt.show()

abs(y_test - regression.predict(x_test)).describe(percentiles = [0.25, 0.5, 0.75, 0.95, 0.99])

"""Creo que esto quiere decir que nuestros resultados son muy acercados. No se si esto puede ser un problema como de overfitting? """


### 6. Mejora del modelo

#### 6.1. Coefficient transformation

# Create a pipeline with a logarithmic transformer, imputation, and linear regression
# Define the column to transform
column_to_transform = ['km_driven']

# Define the transformer to apply to the column
transformer = FunctionTransformer(np.log1p)

# Define the transformers to apply to other columns
other_transformers = []
for feature in features:
    if feature not in column_to_transform:
        other_transformers.append(('imputer', SimpleImputer(strategy='mean'), [feature]))
        other_transformers.append(('scaler', StandardScaler(), [feature]))

# Define the column transformer
column_transformer = ColumnTransformer(
    transformers=[
        ('log_transformer', transformer, column_to_transform),
        *other_transformers
    ])

# Define the pipeline with the column transformer and linear regression model
pipeline = Pipeline([
    ('preprocessing', column_transformer),
    ('linear_regression', LinearRegression())
])

# Fit the model on the training data
pipeline.fit(x_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(x_test)

# Print error
print('Mean Absolute Error:', mean_absolute_error(x_train, transformed_regressor.predict(x_train)))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(mean_squared_error(x_train, transformed_regressor.predict(x_train))))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))



## 8. Exportación del modelo
##### Se debe exportar el modelo hecho anteriormente para poder enviarlo a MotorAlpes para que lo puedan correr
"""

# No sé si hay que incluir lo de preparación de los datos, creo que no

class AntiquityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["year"] = 2020 - X["year"].astype(int)
        return X[["year", "km_driven", "seats", "max_power", "fuel", "transmission"]]

numeric_cols = ["year", "km_driven", "seats", "max_power"]
categorical_cols = ["fuel", "transmission"]

numeric_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("antiquity", AntiquityTransformer())
])

categorical_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(categories='auto', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_preprocessor, numeric_cols),
    ('cat', categorical_preprocessor, categorical_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('regression', LinearRegression())
])

pipeline.fit(x_train, y_train)

"""##### Se debe revisar que el pipeline este funcionando bien"""

y_pred = pipeline.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df['Difference'] = df['Actual'] - df['Predicted']
df['Difference %'] = df['Difference'] / df['Actual'] * 100
df

print('Mean Absolute Error Train:', mean_absolute_error(y_train, pipeline.predict(x_train)))
print('Mean Absolute Error Test:', mean_absolute_error(y_test, pipeline.predict(x_test)))

print('Root Mean Squared Error Train:', np.sqrt(mean_squared_error(y_train, pipeline.predict(x_train))))
print('Root Mean Squared Error Test:', np.sqrt(mean_squared_error(y_test, pipeline.predict(x_test))))

abs(y_test - regression.predict(x_test)).describe(percentiles = [0.25, 0.5, 0.75, 0.95, 0.99])