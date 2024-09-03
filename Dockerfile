FROM python:3.10-slim-bullseye

RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip --progress-bar off

RUN pip install --no-cache-dir -r requirements.txt --progress-bar off

COPY . .

EXPOSE 5100

