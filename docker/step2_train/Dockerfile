# Step 2 — Model training and versioning
# Needs scikit-learn and joblib for ML, plus pandas/pyarrow for data loading.

FROM python:3.12-slim
WORKDIR /app

COPY docker/step2_train/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/__init__.py src/__init__.py
COPY src/data_loader.py src/data_loader.py
COPY src/train.py src/train.py

# SIMULATE_ERROR=1 to demonstrate the induced error scenario
ENV SIMULATE_ERROR=0

ENTRYPOINT ["python", "-m", "src.train"]
