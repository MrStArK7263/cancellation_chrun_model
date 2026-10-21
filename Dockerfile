FROM python:3.13-slim

WORKDIR /app

# install dependencies first (Docker caches this layer, so rebuilds are fast)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy only what the API needs
COPY main.py preprocessing.py ./
COPY model ./model

ENV MODEL_VERSION=1.0.0
ENV MODEL_PATH=model/churn_model_v1.0.0.joblib

# Render gives the port in $PORT (default 8000 locally)
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
