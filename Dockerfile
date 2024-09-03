FROM python:3.10-slim-bullseye

RUN pip install --upgrade pip --progress-bar off

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt --progress-bar off

COPY . .

EXPOSE 5100

