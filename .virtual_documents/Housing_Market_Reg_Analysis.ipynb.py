import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import linear_rainbow, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import LabelEncoder
from scipy import stats


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


positively_correlated_cols = ['price', 'sqft_living', 'floors', 'bedrooms', 'bathrooms']
positively_correlated_df = df[positively_correlated_cols]
sns.pairplot(positively_correlated_df);


plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(df['price']).set_title('Average house price')


fsm_df = df[["sqft_living", "price"]].copy()
fsm_df.dropna(inplace=True)


fsm = ols(formula= "price ~ sqft_living", data=fsm_df)
fsm_results = fsm.fit()


fsm_results.summary()


sns.set_theme(color_codes=True)
f, ax = plt.subplots(figsize=(10, 8))


sns.regplot(x="sqft_living", y="price", data=fsm_df, ax=ax).set_title('Model1 Visualization');
plt.savefig('viz1.png')


rainbow_statistic, rainbow_p_value = linear_rainbow(fsm_results)
print("Rainbow statistic:", rainbow_statistic)
print("Rainbow p-value:", rainbow_p_value)


resid = fsm_results.resid
qq1 = sm.qqplot(resid, line ='45', fit=True, dist=stats.t)


y = fsm_df["price"]

y_hat = fsm_results.predict()


fig2, ax2 = plt.subplots()
ax2.set(xlabel="Price",
        ylabel="Residuals")
ax2.scatter(x=y_hat, y=y_hat-y, color="blue", alpha=0.2);


lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(y-y_hat, fsm_df[["price"]])
print("Lagrange Multiplier p-value:", lm_p_value)
print("F-statistic p-value:", f_p_value)




#necessary?? 

preds = fsm_results.predict()


def plot_predictions(y_true, y_hat):
    fig, axis = plt.subplots()
    axis.scatter(y_true,y_hat,label='Model Output', alpha=.5, edgecolor='black')
    y_equalsx = np.linspace(0,y_true.max())
    axis.set_xlabel('True')
    axis.set_ylabel('Predicted')
    axis.plot(y_equalsx,y_equalsx,label='Predicted = True',color='black')
    #axis.title('fitnes of model explenation')
    axis.legend();


plot_predictions(fsm_df.price,preds)


fitnes of model explenation


negatively_correlated_cols = [
    'bedrooms',
    'condition',
    'yr_built',
    'floors',
    'grade'
]
negatively_correlated_df = df[negatively_correlated_cols]
sns.pairplot(negatively_correlated_df);


model_2_df = df[["sqft_living", "price", "floors"]].copy()
model_2_df.dropna(inplace=True)


model_2 = ols(formula="price ~ sqft_living + floors", data=model_2_df)
model_2_results = model_2.fit()


model_2_results.summary()


sns.pairplot(df,
            x_vars=['sqft_living', 'floors'],
            y_vars=['price'],
             kind='reg').savefig('test1.png')


rainbow_statistic, rainbow_p_value = linear_rainbow(model_2_results)
print("Rainbow statistic:", rainbow_statistic)
print("Rainbow p-value:", rainbow_p_value)


resid = model_2_results.resid
qq2 = sm.qqplot(resid, line ='45', fit=True, dist=stats.t)



y2 = model_2_df["price"]


y_hat_2 = model_2_results.predict()


fig2, ax2 = plt.subplots()
ax2.set(xlabel="Price",
        ylabel="Residuals")
ax2.scatter(x=y_hat_2, y=y_hat_2-y2, color="blue", alpha=0.2);


lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(y2-y_hat_2, model_2_df[["price"]])
print("Lagrange Multiplier p-value:", lm_p_value)
print("F-statistic p-value:", f_p_value)


model_3_df = df[["sqft_living", "price", "floors", 'bedrooms']].copy()
model_3_df.dropna(inplace=True)


model_3 = ols(formula="price ~ sqft_living + floors + bedrooms", data=model_3_df)
model_3_results = model_3.fit()


model_3_results.summary()


sns.pairplot(df,
            x_vars=['sqft_living', 'floors', 'bedrooms'],
            y_vars=['price'],
             kind='reg')


rainbow_statistic, rainbow_p_value = linear_rainbow(model_3_results)
print("Rainbow statistic:", rainbow_statistic)
print("Rainbow p-value:", rainbow_p_value)


resid = model_3_results.resid
qq3 = sm.qqplot(resid, line ='45', fit=True, dist=stats.t)



y3 = model_3_df["price"]


y_hat_3 = model_3_results.predict()


fig3, ax3 = plt.subplots()
ax3.set(xlabel="Price",
        ylabel="Residuals")
ax3.scatter(x=y_hat_3, y=y_hat_3-y3, color="blue", alpha=0.2);


lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(y3-y_hat_3, model_3_df[["price"]])
print("Lagrange Multiplier p-value:", lm_p_value)
print("F-statistic p-value:", f_p_value)



