FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the pipeline once at image build time so a fresh container
# already has data, features, trained models, and a populated DB.
RUN python data/generate_data.py \
    && python features/build_features.py \
    && python models/train_model.py \
    && python db/load_to_db.py

EXPOSE 8000 8501

# Default command runs the API; override for the dashboard service (see docker-compose.yml)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
