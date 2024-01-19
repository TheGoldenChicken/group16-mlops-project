# Base image
FROM python:3.9-slim

EXPOSE 8080
CMD exec uvicorn predictions:app --port 8080 --workers 1 main:app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY fmnist/ fmnist/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
<<<<<<< HEAD
=======
# RUN pip install . --no-deps --no-cache-dir
>>>>>>> a082deeb8a2c87cbe949d7908b984b257b57b849

ENTRYPOINT ["python", "-u", "fmnist/predict_model.py"]