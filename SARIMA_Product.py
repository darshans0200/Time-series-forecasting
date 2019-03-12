#!/usr/bin/env python
# coding: utf-8

# # SARIMA
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import matplotlib
import pickle
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

def timeseries_train_test_split(paneer, test_size):
    """
        Perform train-test split with respect to time series structure
    """
    
    # get the index after which test set starts
    test_index = int(len(paneer)*(1-test_size))
    
    X_train = paneer.iloc[:test_index]
    X_test = paneer.iloc[test_index:]

    
    return X_train, X_test

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


df = pd.read_csv('C:\\Users\\Darshan\\Downloads\\Misty\\revised\\name1.csv', sep=',')
df['Stime'] = pd.to_datetime(df['Stime'],format="%d-%m-%Y")


df=df.drop(['SALESEXECUTIVEID','ASMID','ZONALHEADID','SALESHEADID'],axis=1)

cols = list(pd.read_csv('C:\\Users\\Darshan\\Downloads\\Misty\\revised\\product1.csv', nrows =1))
print(cols)

prod=pd.read_csv('C:\\Users\\Darshan\\Downloads\\Misty\\revised\\product1.csv', sep=',',usecols =[i for i in cols if i != ('description')])
prod=prod.drop(['Createdt', 'CreateID','productId', 'productCategory','Updatedt','activeProduct','minimumRate','maximumRate','photo','updateID','HSNCode'], axis=1)

m=pd.merge(df, prod, on='Item_number_ID')

products=m.Product.unique()
#products=m.loc[(['Product']!='MILK')]

print (products)
for item in products:
    print (item)
    if item=='MILK':
        continue
    paneer=m.loc[(m['Product']==item)]
    cols = ['Id', 'Year', 'Month', 'Day', 'Item_number_ID',
            'Customer_Account_ID', 'regionID', 'Amount', 'productName',
            'Product', 'Sub_Product', 'Item_name', 'weight', 'weightUnit',
            'shelfLife', 'purchaseRate', 'retailRate', 'GSTPercentage']
    paneer.drop(cols, axis=1, inplace=True)
    paneer=paneer.resample('D', on='Stime').sum()

    X_train, X_test = timeseries_train_test_split(paneer, test_size=0.3)
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(X_train,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
    
                results = mod.fit()
    
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    

    mod = sm.tsa.statespace.SARIMAX(X_train,
                                    order=(1, 1, 1),
                                    seasonal_order=(0, 1, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()
    
  
    pickle.dump(results, open(item+"-model_SARIMA.pkl", "wb"))

    #print(results.summary().tables[1])

#Test set prediction

#start_index = '2018-11-01'
#end_index = '2019-01-31'
#forecast_test = results.predict(start=start_index, end=end_index)

#forecast_test.head()

#forecast_test=forecast_test.to_frame(name=None)

#forecast_test=forecast_test.rename(columns = {0:'Quantity_predicated'})

#index = pd.date_range(start="2018-11-01", end="2019-01-31")

#forecast_test['Stime']=index

#forecast_test.set_index('Stime', inplace=True)



'''
#test set error

forecast_test_error = X_test['Quantity']-forecast_test['Quantity_predicated']

mean_forecast_test_error = np.mean(forecast_test_error)
print(mean_forecast_test_error)

mean_absolute_test_error = np.mean( np.abs(forecast_test_error) )
print(mean_absolute_test_error)

mean_squared_test_error = np.mean(forecast_test_error*forecast_test_error)
print(mean_squared_test_error)

rmse_test = np.sqrt(mean_squared_test_error)
print(rmse_test)

def mean_absolute_percentage_test_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


MAPE= mean_absolute_percentage_test_error(X_test['Quantity'],forecast_test['Quantity_predicated'])
print(MAPE)

X_test.shape,forecast_test.shape

pjme_test=pd.merge(X_test,forecast_test, left_index=True, right_index=True)

MAPE= mean_absolute_percentage_train_error(pjme_test['Quantity'],pjme_test['Quantity_predicated'])
print(MAPE)

pjme_test.plot(figsize=(15, 6))
plt.show()

#another method for predict
#pred_uc = results.get_forecast(steps=92)
#pred_ci = pred_uc.conf_int()

start_index = '2018-04-01'
end_index = '2018-10-31'
forecast_train = results.predict(start=start_index, end=end_index)

forecast_train=forecast_train.to_frame(name=None)

forecast_train=forecast_train.rename(columns = {0:'Quantity_predicated'})
forecast_train.head()
forecast_train.tail()

#Train set prediction

index = pd.date_range(start="2018-04-01", end="2018-10-31")
forecast_train['Stime']=index
forecast_train.set_index('Stime', inplace=True)
forecast_train.head()
X_train.head()

#Train set error

forecast_train_error = X_train['Quantity']-forecast_train['Quantity_predicated']
mean_forecast_train_error = np.mean(forecast_train_error)
print(mean_forecast_train_error)

mean_absolute_train_error = np.mean( np.abs(forecast_train_error) )
print(mean_absolute_train_error)
mean_squared_train_error = np.mean(forecast_train_error*forecast_train_error)
print(mean_squared_test_error)


rmse_train = np.sqrt(mean_squared_train_error)
print(rmse_test)

def mean_absolute_percentage_train_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
MAPE= mean_absolute_percentage_train_error(X_train['Quantity'],forecast_train['Quantity_predicated'])
print(MAPE)

pjme_train=pd.merge(X_train,forecast_train, left_index=True, right_index=True)

MAPE= mean_absolute_percentage_train_error(pjme_train['Quantity'],pjme_train['Quantity_predicated'])
print(MAPE)

pjme_train.plot(figsize=(15, 6))
plt.show()

all = pd.concat([pjme_train,pjme_test], sort=False)
all.head()
all.plot(figsize=(15, 6))
plt.show()
'''


'''
#Forecast month on unseen data

start_index = '2019-01-28'
end_index = '2019-02-23'
forecast = results.predict(start=start_index, end=end_index)

forecast=forecast.to_frame(name=None)
forecast=forecast.rename(columns = {0:'Quantity_predicated'})
forecast.head()

index = pd.date_range(start="2019-01-28", end="2019-02-23")
forecast['Stime']=index
forecast.set_index('Stime', inplace=True)

forecast.head()

all_l = pd.concat([pjme_train,pjme_test,forecast], sort=False)

all_l.tail()

all_l.plot(figsize=(15, 6))
plt.show()
'''