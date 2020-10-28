import pytest
from fastapi.testclient import TestClient
from main import app
from ml import predict_test, predict, pre_process
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

client = TestClient(app)

@pytest.mark.us001ts001
def test_request_method():
    ## Should not allow any other request method other than POST
    response = client.get("/predict")
    assert response.status_code == 405
    assert response.json() == {"detail": "Method Not Allowed"}
    response = client.put("/predict")
    assert response.status_code == 405
    assert response.json() == {"detail": "Method Not Allowed"}
    response = client.delete("/predict")
    assert response.status_code == 405
    assert response.json() == {"detail": "Method Not Allowed"}

@pytest.mark.us001ts001
def test_number_of_endpoints():
    ## The number of URLs we should have is 5 including the docs and default URLs of FastAPI
    url_list = [
        {'path': route.path, 'name': route.name}
        for route in app.routes
    ]
    assert len(url_list) == 5 

@pytest.mark.us001ts001
def test_input_format():
    ## Test the expected format
    response = client.post(
        "/predict",
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
    ## Test cases for missing keys
    response = client.post(
        "/predict",
        json={
            "Store": 1111,
            "Date": "2014-07-10",
            "Customers": 410,
            "Open": 1,
            "Promo": 0,
            "StateHoliday": "0",
            "SchoolHoliday": 1
        },
    )
    assert response.status_code == 422
    result_set = response.json()
    assert result_set["detail"][0]["msg"] == "field required"
    response = client.post(
        "/predict",
        json={
            "Store": 1111,
            "Date": "2014-07-10",
            "Customers": 410,
            "Open": 1,
            "SchoolHoliday": 1
        },
    )
    assert response.status_code == 422
    result_set = response.json()
    assert result_set["detail"][0]["msg"] == "field required"
    ## Test for completely different json
    response = client.post(
        "/predict",
        json={
            "dog": 1,
            "cat": 0
        },
    )
    assert response.status_code == 422
    result_set = response.json()
    assert result_set["detail"][0]["msg"] == "field required"

@pytest.mark.us001ts002_3
def test_rmse_and_output():
    ## Pre-processing steps
    store = pd.read_csv("../datasets/store.csv")
    df = pd.read_csv("../datasets/train.csv",parse_dates=[2])
    store.fillna(0, inplace=True)
    merged = pd.merge(df, store, on='Store')
    merged = merged.sort_values(['Date'],ascending = False)
    merged = merged[(merged.Open != 0)&(merged.Sales >0)]
    merged.loc[:,"StateHoliday"] = df.StateHoliday.apply(lambda x: '0' if x == 0 else x)
    X = merged[['Date', 'Store', 'DayOfWeek', 'Customers', 'Promo', 'StateHoliday', 'SchoolHoliday',
       'StoreType', 'Assortment', 'CompetitionDistance',
       'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
       'Promo2SinceWeek', 'Promo2SinceYear']].copy()
    Y = np.log1p(merged["Sales"])
    X.loc[:,"Month"] = X.Date.dt.month
    X.loc[:,"Year"] = X.Date.dt.year
    X.loc[:,"Day"] = X.Date.dt.day
    X.loc[:,'WeekOfYear'] = X.Date.dt.isocalendar().week.astype("int64")
    X.drop(columns = ['Date'], inplace=True)
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    X.StoreType.replace(mappings, inplace=True)
    X.Assortment.replace(mappings, inplace=True)
    X.StateHoliday.replace(mappings, inplace=True)
    X.loc[:,'CompetitionOpen'] = 12 * (X.Year - X.CompetitionOpenSinceYear) + \
        (X.Month - X.CompetitionOpenSinceMonth)
    X.loc[:,'CompetitionOpen'] = X['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)
    X.loc[:,'PromoOpen'] = 12 * (X.Year - X.Promo2SinceYear) + \
        (X.WeekOfYear - X.Promo2SinceWeek) / 4.0
    X.loc[:,'PromoOpen'] = X['PromoOpen'].apply(lambda x: x if x > 0 else 0)

    # Use the last 25% of the data as a validation set
    train_percentage = 0.25
    train_size = int(df.shape[0] * train_percentage)
    X_val = X[:train_size] 
    y_val = Y[:train_size]

    # Load the model
    y_pred = predict_test(X_val)
    rmse_result = mean_squared_error(np.expm1(y_val), np.expm1(y_pred), squared=False)

    assert rmse_result < 2000 # RMSE should be less than 2000
    assert type(y_pred[0]) == np.float64 # Model output should be numeric/float
test_rmse_and_output()
@pytest.mark.us001ts004
def test_api_response():
    ## Test the expected format
    response = client.post(
        "/predict",
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
