import pytest
import requests
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

@pytest.mark.us001ts005
def test_prod_request():
    ## Check whether prod is accepting the correct format and returning the correct output
    response = requests.post(
        "https://renzorossmann.herokuapp.com/predict",
        json={
            "Store": 1111,
            "DayOfWeek": 4,
            "Date": "2014-07-10",
            "Customers": 410,
            "Open": 1,
            "Promo": 0,
            "StateHoliday": "0",
            "SchoolHoliday": 1
        },
    )
    assert response.status_code == 200
    result_set = response.json()
    assert len(result_set.keys()) == 1
    assert list(result_set.keys())[0] == "sales"
    assert type(result_set[list(result_set.keys())[0]]) == float

@pytest.mark.us001ts005
def test_prod_rmse():
    ## Send 10 random samples from our dataset to our api and check the rmse
    df = pd.read_csv("../datasets/train.csv",parse_dates=[2]).sample(10)
    y_val = df["Sales"]
    y_pred = []
    for i in range(10):
        print(int(df.iloc[i,:]["Open"]))
        response = requests.post(
            "https://renzorossmann.herokuapp.com/predict",
            json={
                "Store": int(df.iloc[i,:]["Store"]),
                "DayOfWeek": int(df.iloc[i,:]["DayOfWeek"]),
                "Date": str(df.iloc[i,:]["Date"]),
                "Customers": int(df.iloc[i,:]["Customers"]),
                "Open": int(df.iloc[i,:]["Open"]),
                "Promo": int(df.iloc[i,:]["Promo"]),
                "StateHoliday": str(df.iloc[i,:]["StateHoliday"]),
                "SchoolHoliday": int(df.iloc[i,:]["SchoolHoliday"])
            },
        )
        result_set = response.json()
        y_pred.append(result_set["sales"])
    rmse_result = mean_squared_error(y_val, y_pred, squared=False)
    assert rmse_result < 2000