from joblib import load
import numpy as np
import pandas as pd
from fastapi.encoders import jsonable_encoder
import os
import boto3

"""
This file contains the functions related to the ML model.
"""

def download_from_s3():
    """
    Function for downloading the most recent model in AWS s3
    """
    s3 = boto3.resource(
        service_name='s3',
        region_name=os.environ["AWS_DEFAULT_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    s3.Bucket('rossmann-mynt').download_file(Key='models/model.joblib', Filename='../model/model.joblib')

## Check working directory and Set if not valid (FOR DOCKER)
if "main.py" not in os.listdir('.'):
    print(os.getcwd())
    print("Settings Working Directory to app/")
    os.chdir("/app/app/")

## Load the model and store data
if os.path.exists('../model/model.joblib'):
    clf = load('../model/model.joblib') 
else:
    download_from_s3() # Download the model from s3 if not found
store = pd.read_csv("../datasets/store.csv")
store.fillna(0, inplace=True)

def pre_process(data):
    sample = pd.DataFrame(data.dict(), index=[0])
    ## PRE-PROCESSING STEPS
    X = sample.merge(store, on='Store', how='left')
    X.loc[:,"Date"] = X["Date"].astype('datetime64[ns]')
    X["Month"] = X.Date.dt.month
    X["Year"] = X.Date.dt.year
    X["Day"] = X.Date.dt.day
    X['WeekOfYear'] = X.Date.dt.isocalendar().week
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    X.StoreType.replace(mappings, inplace=True)
    X.Assortment.replace(mappings, inplace=True)
    X.StateHoliday.replace(mappings, inplace=True)
    X['CompetitionOpen'] = 12 * (X.Year - X.CompetitionOpenSinceYear) + \
        (X.Month - X.CompetitionOpenSinceMonth)
    X['CompetitionOpen'] = X['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)
    X['PromoOpen'] = 12 * (X.Year - X.Promo2SinceYear) + \
        (X.WeekOfYear - X.Promo2SinceWeek) / 4.0
    X['PromoOpen'] = X['PromoOpen'].apply(lambda x: x if x > 0 else 0)
    
    X.drop(columns = ['Date','Open', 'Customers', 'Open', 'PromoInterval'], inplace=True)
    # Store	DayOfWeek	Promo	StateHoliday	SchoolHoliday	StoreType	Assortment	CompetitionDistance	CompetitionOpenSinceMonth	CompetitionOpenSinceYear	Promo2	Promo2SinceWeek	Promo2SinceYear	Month	Year	Day	WeekOfYear	CompetitionOpen	PromoOpen
    # pd.Series(data.dict())
    return X.to_numpy()

def predict(data):
    y_pred = clf.predict(data)
    return np.expm1(y_pred[0])

def predict_test(data):
    y_pred = clf.predict(data)
    return y_pred
# Define eval metrics
def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)