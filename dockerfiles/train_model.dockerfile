# Base image
FROM python:3.9-slim

EXPOSE 8080

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY fmnist/ fmnist/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
# RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "fmnist/train_model.py"]