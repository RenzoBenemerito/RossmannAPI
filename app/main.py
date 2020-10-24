from fastapi import FastAPI
from pydantic import BaseModel

## DATA MODEL
## This describes the format of the json that our API expects. All fields are required
class Payload(BaseModel):
    Store: int
    DayOfWeek: int
    Date: str
    Customers: int
    Open: int
    Promo: int
    StateHoliday: str
    SchoolHoliday: int


app = FastAPI()

@app.post("/predict")
async def predict_sales(payload: Payload):
    """
    This endpoint accepts a POST request. The format of the data it accepts is presented in the Payload data model.
    Upon receiving JSON data, it inputs the data to a model. The output prediction of the model is returned as a
    JSON response
    Authored by: Renzo Benemerito

    input: JSON, data model -> Payload
    output: JSON format { sales: int }
    DOCS URL: /docs#/default/predict_sales_predict_post
    """
    return payload