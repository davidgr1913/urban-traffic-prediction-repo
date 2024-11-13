FROM python:3.10-slim

WORKDIR /app

COPY /requirements.txt /app/

COPY /models /app/models


RUN pip install --no-cache-dir -r requirements.txt

RUN pip install fastapi uvicorn

COPY app.py /app/

COPY src /app/src

EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]