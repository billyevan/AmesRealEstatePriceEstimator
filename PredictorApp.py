import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor

import matplotlib
matplotlib.rcParams["figure.figsize"] = (20, 10)

df = pd.read_csv("./house_prices.csv")
df1 = df[['LotArea', 'Neighborhood', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'KitchenAbvGr', 'TotalBsmtSF', 'SalePrice']]
df1['Bathroom'] = df1.loc[:, ['HalfBath','FullBath']].sum(axis=1)
df2 = df1[df1.columns[~df1.columns.isin(['HalfBath','FullBath'])]]
df3 = df2.copy()
df3['AreaAboveGr'] = df3.LotArea - df3.TotalBsmtSF
df4 = df3.copy()
df4['PriceSqFt'] = df4['SalePrice'] / df4['LotArea']

def remove_pricesqft_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('Neighborhood'):
        m = np.mean(subdf.PriceSqFt)
        st = np.std(subdf.PriceSqFt)
        reduced_df = subdf[ (subdf.PriceSqFt > (m-st)) & (subdf.PriceSqFt <= (m+st)) ]
        df_out = pd.concat([df_out, reduced_df],ignore_index=True)
    return df_out

df5 = remove_pricesqft_outliers(df4)

df_a = df5.copy()
df_a = df_a.drop(['TotalBsmtSF', 'AreaAboveGr', 'PriceSqFt'], axis='columns')

dummies_nghbd = pd.get_dummies(df_a.Neighborhood)

df_n = pd.concat([df_a,dummies_nghbd],axis='columns')

df_n2 = df_n.drop('Neighborhood', axis='columns')

X = df_n2.drop('SalePrice', axis='columns')
y = df_n2.SalePrice
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)
lr_clf.score(X_test, y_test)

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv)

def predict_price(neighborhood, area, nb_bed, nb_bath, nb_kitch):
    loc_index = np.where(X.columns == neighborhood)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = area
    x[1] = nb_bed
    x[2] = nb_kitch
    x[3] = nb_bath
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

st.title("Real-Estate Price Estimator")
st.header("With the help of AI and Machine Learning (and some Data Science magic), find the best estimate for your real-estate in Ames, Iowa.")

col1, col2 = st.columns(2)

list_neighborhoods = ('Bloomington Heights',
'Bluestem',
'Briardale',
'Brookside',
'Clear Creek',
'College Creek',
'Crawford',
'Edwards',
'Gilbert',
'Iowa DOT and Rail Road',
'Meadow Village',
'Mitchell',
'North Ames',
'Northridge',
'Northpark Villa',
'Northridge Heights',
'Northwest Ames',
'Old Town',
'South & West of Iowa State University',
'Sawyer',
'Sawyer West',
'Somerset',
'Stone Brook',
'Timberland',
'Veenker')

area_sqft = 0

with col1:
    area_sqft = st.number_input(
        "Total area (in square feet)",
        min_value=0, max_value=9999999, step=1,value=1000
    )
    neigh =  st.selectbox(
        "Neighborhood",
        list_neighborhoods
    )

if neigh == 'Bloomington Heights':
    neigh = 'Blmngtn'
elif neigh == 'Bluestem':
    neigh = 'Blueste'
elif neigh == 'Briardale':
    neigh = 'BrDale'
elif neigh == 'Brookside':
    neigh = 'BrkSide'
elif neigh == 'Clear Creek':
    neigh = 'ClearCr'
elif neigh == 'College Creek':
    neigh = 'CollgCr'
elif neigh == 'Crawford':
    neigh = 'Crawfor'
elif neigh == 'Edwards':
    neigh = 'Edwards'
elif neigh == 'Gilbert':
    neigh = 'Gilbert'
elif neigh == 'Iowa DOT and Rail Road':
    neigh = 'IDOTRR'
elif neigh == 'Meadow Village':
    neigh = 'MeadowV'
elif neigh == 'Mitchell':
    neigh = 'Mitchel'
elif neigh == 'North Ames':
    neigh = 'Names'
elif neigh == 'Northridge':
    neigh = 'NoRidge'
elif neigh == 'Northpark Villa':
    neigh = 'NPkVill'
elif neigh == 'Northridge Heights':
    neigh = 'NridgHt'
elif neigh == 'Northwest Ames':
    neigh = 'NWAmes'
elif neigh == 'Old Town':
    neigh = 'OldTown'
elif neigh == 'South & West of Iowa State University':
    neigh = 'SWISU'
elif neigh == 'Sawyer':
    neigh = 'Sawyer'
elif neigh == 'Sawyer West':
    neigh = 'SawyerW'
elif neigh == 'Somerset':
    neigh = 'Somerst'
elif neigh == 'Stone Brook':
    neigh = 'StoneBr'
elif neigh == 'Timberland':
    neigh = 'Timber'
elif neigh == 'Veenker':
    neigh = 'Veenker'


with col2:
    nb_bedroom = st.slider('Number of bedroom(s)', 1, 6, 2)
    nb_bathroom = st.slider('Number of bathroom(s)', 1, 6, 2)
    nb_kitchen = st.slider('Number of kitchen(s)', 0, 3, 1)

price = predict_price(neigh, area_sqft, nb_bedroom, nb_kitchen, nb_bathroom)

if st.button('Calculate'):
    if (area_sqft > 0):
        st.write('The estimated price is $', abs(round(price)))
    else:
        st.error('Cannot calculate with such area value, please recheck your inputs!')