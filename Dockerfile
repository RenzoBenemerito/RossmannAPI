FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR /app/app/

COPY . /app/

RUN pip install -r /app/requirements.txt
