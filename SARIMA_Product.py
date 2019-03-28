"""
Created on Wed Mar  6 09:36:29 2019

@author: Darshan
"""

#!/usr/bin/env python
# coding: utf-8

# # SARIMA
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
import pickle
import statsmodels.api as sm 
import mysql.connector 
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

mydb=mysql.connector.connect(host="**.***.***.***", user="***", passwd="***", database="***")

def mean_absolute_percentage_test_all_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


df = pd.read_sql('select * from raw_sales_dump', mydb)
df['Stime'] = pd.to_datetime(df['Stime'],format="%Y-%m-%d")
df=df.drop(['SALESEXECUTIVEID','ASMID','ZONALHEADID','SALESHEADID'],axis=1)


prod=pd.read_sql('select * from product',mydb)
prod=prod.drop(['Createdt', 'CreateID','productId', 'productCategory','Updatedt','activeProduct','minimumRate','maximumRate','photo','updateID','HSNCode','description','GSTPercentage'], axis=1)

m=pd.merge(df, prod, on='Item_number_ID')
products=m.Product.unique()
print (products)

for item in products:
    print (item)
    if item=='MILK':
        continue
    paneer=m.loc[(m['Product']==item)]
    cols = ['Id', 'Year', 'Month', 'Day', 'Item_number_ID',
            'Customer_Account_ID', 'regionID', 'Amount', 'productName',
            'Product', 'Sub_Product', 'Item_name', 'weight', 'weightUnit',
            'shelfLife', 'purchaseRate', 'retailRate']
    paneer.drop(cols, axis=1, inplace=True)
    paneer=paneer.resample('D', on='Stime').sum()

 
    p = q = range(0, 5)
    d = range(0,1)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    # Set variables to populate
    lowest_aic = None
    lowest_parm = None
    lowest_param_seasonal = None
    # GridSearch the hyperparameters of p, d, q and P, D, Q, m
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(paneer,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
    
                results = mod.fit()
                
                # Store results
                current_aic = results.aic
                # Set baseline for aic
                if (lowest_aic == None):
                    lowest_aic = results.aic
                # Compare results
                if (current_aic <= lowest_aic):
                    lowest_aic = current_aic
                    lowest_parm = param
                    lowest_param_seasonal = param_seasonal
    
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
            
    print('The best model is: SARIMA{}x{} - AIC:{}'+item.format(lowest_parm, lowest_param_seasonal, lowest_aic))     

    mod = sm.tsa.statespace.SARIMAX(paneer,
                                    order=lowest_parm,
                                    seasonal_order=lowest_param_seasonal,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()
    
  
    pickle.dump(results, open(item+"-model_SARIMA.pkl", "wb"))

    
    
#it's reference,can be removed    
    start_index = '2018-04-01'
    end_index = '2019-01-31'
    forecast_test_all = results.predict(start=start_index, end=end_index)
    forecast_test_all=forecast_test_all.to_frame(name=None)
    forecast_test_all=forecast_test_all.rename(columns = {0:'Quantity_predicated'})
    index = pd.date_range(start="2018-04-01", end="2019-01-31")
    forecast_test_all['Stime']=index
    forecast_test_all.set_index('Stime', inplace=True)
    forecast_test_all_error = paneer['Quantity']-forecast_test_all['Quantity_predicated']
    mean_forecast_test_all_error = np.mean(forecast_test_all_error)
    print(mean_forecast_test_all_error)
    mean_absolute_test_all_error = np.mean( np.abs(forecast_test_all_error) )
    print(mean_absolute_test_all_error)
    mean_squared_test_all_error = np.mean(forecast_test_all_error*forecast_test_all_error)
    print(mean_squared_test_all_error)
    rmse_test_all = np.sqrt(mean_squared_test_all_error)
    print(rmse_test_all)

    MAPE= mean_absolute_percentage_test_all_error(paneer['Quantity'],forecast_test_all['Quantity_predicated'])
    print('MAPE'+MAPE)
    pjme_all_test=pd.merge(paneer,forecast_test_all, left_index=True, right_index=True)
    MAPE= mean_absolute_percentage_test_all_error(pjme_all_test['Quantity'],pjme_all_test['Quantity_predicated'])
    print('MAPE'+MAPE)
    
