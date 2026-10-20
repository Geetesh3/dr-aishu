# Dr. Aishu — production image.
# Includes Tesseract so the image-analysis endpoint works out of the box.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DR_AISHU_DATA_DIR=/data

# Tesseract is the only system dependency (OCR of lab reports and prescriptions).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so this layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY static ./static
COPY templates ./templates
COPY notebooks ./notebooks
COPY data/models ./data/models

# Writable location for the SQLite case database and API keys. Mount a volume
# here on hosts that offer persistent disks; otherwise it resets on redeploy.
RUN mkdir -p /data
VOLUME ["/data"]

# Most PaaS providers inject PORT; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

# Render/Fly set PORT at runtime; the shell form expands it.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
