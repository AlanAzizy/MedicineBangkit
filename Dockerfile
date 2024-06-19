FROM python:3.12

WORKDIR /

COPY ./requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY ./App /App

CMD ["uvicorn", "App.main:app", "--host", "0.0.0.0", "--port", "8000"]