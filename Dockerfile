FROM python:3.10-slim

WORKDIR /app

COPY requirements/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY src/ src/
COPY config/ config/
COPY data/artifacts/features/05_preprocessor.joblib data/artifacts/features/05_preprocessor.joblib
COPY data/artifacts/model/06_logistic_regression_balanced.joblib data/artifacts/model/06_logistic_regression_balanced.joblib

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
