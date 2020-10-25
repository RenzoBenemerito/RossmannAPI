from fastapi import FastAPI
from models import Payload
from ml import pre_process, predict

app = FastAPI()

@app.post("/predict", name="predict_rossmann_sales")
async def predict_rossmann_sales(payload: Payload):
    """
    This endpoint accepts a POST request. The format of the data it accepts is presented in the Payload data model.
    Upon receiving JSON data, it inputs the data to a model. The output prediction of the model is returned as a
    JSON response
    Authored by: Renzo Benemerito

    input: JSON, data model -> Payload
    output: JSON format { sales: int }
    DOCS URL: /docs#/default/predict_sales_predict_post
    """
    data = pre_process(payload)
    sales = round(predict(data),3)
    result_set = { "sales": sales }
    return result_set