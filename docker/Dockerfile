FROM python:3.8-slim-buster

COPY requirements.txt /

RUN pip install -r /requirements.txt
RUN python -m spacy download en_core_web_sm

# Create a working directory
RUN mkdir /app
WORKDIR /app

ENV PYTHONPATH=/app