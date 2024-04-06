import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

df1 = pd.read_csv('/kaggle/input/bangalore-house-price-prediction/Bengaluru_House_Data.csv')

df1.head()

df1.shape

df1.groupby('area_type')['area_type'].count()

df1.drop(['area_type', 'society','balcony','availability'], axis=1, inplace=True)

df1.head()

df1.isnull().sum()

df1.dropna(axis=0, inplace=True)

df1.isnull().sum()

df1.shape

df1['size'].unique()

df1['bhk'] = df1['size'].apply(lambda x: int(x.split(' ')[0]))

df1.drop(["size"], axis=1, inplace=True)

df1.head()

df1['bhk'].unique()

df1[df1['bhk']>20]

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df1['total_sqft'] = df1['total_sqft'].apply(convert_sqft_to_num)
df1.head()

df1.isna().sum()
df1.dropna(axis=0, inplace=True)

df1['price_per_sqft']  = df1['price']*100000/df1['total_sqft']
df1.head()

len(df1['location'].unique())

df1["location"] = df1["location"].apply(lambda x: x.strip())

location_stats = df1.groupby('location')['location'].count().sort_values(ascending=False)
location_stats

len(location_stats[location_stats<=10])

location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10

df1["location"] = df1["location"].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

dumies = pd.get_dummies(df1["location"])
dumies.head(3)

df1 = pd.concat([df1, dumies.drop('other',axis = 'columns')], axis='columns')
df1.head()

df1 = df1.drop(['location'], axis='columns')
df1.head()

X = df1.drop('price', axis='columns')
X.head()

y = df1["price"]
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=10)

from sklearn.linear_model import LinearRegression
lr_clf=  LinearRegression()
lr_clf.fit(X_train, y_train)lr_clf.score(X_test,y_test)

lr_clf.score(X_test,y_test)