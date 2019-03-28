
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:36:29 2019

@author: Darshan
"""
import pickle
import sys
import pandas as pd
import warnings
#import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
#plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import datetime
import mysql.connector

'''
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
'''
Product=sys.argv[1]
start_date=sys.argv[2]
end_date=sys.argv[3]
version ='v1'
print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))


Product=Product.upper()
modelfile=Product+'-model_SARIMA.pkl'
SARIMA_model_pkl = open(modelfile, 'rb')
pickle_model = pickle.load(SARIMA_model_pkl)
print ("Loaded SARIMA model :: ", pickle_model)
Forecast = pickle_model.predict(start=start_date, end=end_date)
Forecast=Forecast.to_frame(name=None)
Forecast=Forecast.rename(columns = {0:'Quantity_predicated'})
index1 = pd.date_range(start=start_date, end=end_date)
Forecast['Stime']=index1
print(Forecast.head())

#prod 
mydb=mysql.connector.connect(host="**.***.***.***", user="***", passwd="***", database="***")


mycursor = mydb.cursor()


dict=[] #date, quantity, version, product
for index,row in Forecast.iterrows():
    dict.append((index,row['Quantity_predicated'],version,Product))
    print (index,row['Quantity_predicated'])
    
sql1="INSERT INTO ml_daily_estimation (Date, quantity,version,product) VALUES (%s, %s,%s,%s)"
mycursor.executemany(sql1, dict)
mydb.commit()

Forecast=Forecast.resample('M', on='Stime').sum()
print('monthly estimations')
print(Forecast)
year,month,day=start_date.split('-')
yrmonth=month+'/'+year
dict2=[] #date, quantity, version, product
Forecast_month=Forecast.iloc[0]
print('first row estmation',Forecast_month)


oldstart=(datetime.datetime.strptime(start_date,"%Y-%m-%d").date()) -datetime.timedelta(days=183)
#print(oldstart)
oldend=(datetime.datetime.strptime(start_date,"%Y-%m-%d").date()) -datetime.timedelta(days=1)
#print(oldend)
Forecast_old = pickle_model.predict(start=oldstart, end=oldend)
Forecast_old=Forecast_old.to_frame(name=None)
Forecast_old=Forecast_old.rename(columns = {0:'Quantity_predicated'})
index = pd.date_range(start=oldstart, end=oldend)
Forecast_old['Stime']=index

Forecast_old=Forecast_old.resample('M', on='Stime').sum()
print(Forecast_old)

#Last six months average
average=Forecast_old['Quantity_predicated'].mean()
print(average)

target=(Forecast_month-average)/average
print(target)

percent=target['Quantity_predicated']
print(percent)

month_tgt=Forecast_month['Quantity_predicated']
val2 = (yrmonth, Product,'All',percent,month_tgt)
print(val2)

sql2 = "INSERT INTO ml_monthly_estimation (YearMonth,Product,item_number_id,target_percent,target_quantity) VALUES (%s,%s,%s,%s,%s)"
mycursor.execute(sql2, val2)
mydb.commit()

mycursor.close()
mydb.close()

SARIMA_model_pkl.close()
