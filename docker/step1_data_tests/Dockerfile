# Step 1 — Data quality tests
# Only needs pandas, pyarrow, and pytest. No ML dependencies.

FROM python:3.12-slim
WORKDIR /app

COPY docker/step1_data_tests/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/data_loader.py src/data_loader.py
COPY src/__init__.py src/__init__.py
COPY tests/ tests/
COPY conftest.py .

ENTRYPOINT ["python", "-m", "pytest", "tests/", "-v"]
