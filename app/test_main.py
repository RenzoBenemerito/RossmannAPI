import pytest
from fastapi.testclient import TestClient

from main import app

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
    result_set = response.json()
    assert len(result_set.keys()) == 1
    assert list(result_set.keys())[0] == "sales"
    assert type(result_set[list(result_set.keys())[0]]) == float
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
    ## TESTING CI DEV
    