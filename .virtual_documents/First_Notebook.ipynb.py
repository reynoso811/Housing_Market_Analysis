import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import linear_rainbow, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('data/kc_house_data.csv')


df.head()


df.describe()


list(df.columns)


df.isna().sum()


df.dtypes


df.shape


df = df.set_index('price').reset_index()


df.head() 


correraltion = df.corr()
sns.heatmap(correraltion);



# The mask is not necessary, but corr() has duplicate values on either side of the diagonal
mask = np.triu(np.ones_like(correraltion, dtype=np.bool))

correraltion = correraltion.sort_values(by='price', ascending=False)

fig1, ax1 = plt.subplots(figsize=(11, 9))
sns.heatmap(correraltion, mask=mask, ax=ax1, cmap="viridis");


#drop ID column and reset index? whats the purpose of ;?


positively_correlated_cols = ['price', 'sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']
positively_correlated_df = df[positively_correlated_cols]
sns.pairplot(positively_correlated_df);


fsm_df = df[["sqft_living", "price"]].copy()
fsm_df.dropna(inplace=True)


fsm = ols(formula= "price ~ sqft_living", data=fsm_df)
fsm_results = fsm.fit()


fsm_results.summary()


preds = fsm_results.predict()





def plot_predictions(y_true, y_hat):
    fig, axis = plt.subplots()
    axis.scatter(y_true,y_hat,label='Model Output', alpha=.5, edgecolor='black')
    y_equalsx = np.linspace(0,y_true.max())
    axis.set_xlabel('True')
    axis.set_ylabel('Predicted')
    axis.plot(y_equalsx,y_equalsx,label='Predicted = True',color='black')
    axis.legend();


plot_predictions(fsm_df.price,preds)



