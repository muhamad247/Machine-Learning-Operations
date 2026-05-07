# Step 3 — Model robustness validation
# Needs scikit-learn (for loading the model), joblib, pandas, pyarrow, numpy.

FROM python:3.12-slim
WORKDIR /app

COPY docker/step3_validate/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/__init__.py src/__init__.py
COPY src/data_loader.py src/data_loader.py
COPY src/train.py src/train.py
COPY src/validate.py src/validate.py

ENTRYPOINT ["python", "-m", "src.validate"]
