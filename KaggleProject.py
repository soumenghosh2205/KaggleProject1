import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

df = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')

#handle missing data
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Year of Record'].fillna(df['Year of Record'].mean(), inplace=True)

df = df.fillna({"Gender": "missing", "University Degree":"missing","Hair Color": "missing", 'Profession':'missing' })

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
IQR

df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

cat_cols = ['Gender','Profession','University Degree', 'Hair Color', 'Country']

encoder = ce.BinaryEncoder(cols=cat_cols)
df = encoder.fit_transform(df)

X = df.drop(columns='Income in EUR')  #independent columns
y = df['Income in EUR']    #target column

model = LinearRegression()

"""
#split and train
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rms = sqrt(mean_squared_error(y_test,y_pred))
rms
#model.score(X_test, y_test)
"""

#"""
#train using whole data
model.fit(X,y)
#"""


#Test file
df = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')

#handle missing data
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Year of Record'].fillna(df['Year of Record'].mean(), inplace=True)

df.fillna({"Gender": "missing", "University Degree":"missing","Hair Color": "missing", 'Profession':'missing' })

df = encoder.fit_transform(df)

X = df.drop(columns='Income')  

y_pred = model.predict(X)
df['Income']= y_pred

df.to_csv("predicted.csv")


