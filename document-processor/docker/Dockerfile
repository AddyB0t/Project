FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    rustc \
    cargo \
    pkg-config \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p storage/uploads storage/processed storage/extracted_text storage/extracted_images storage/thumbnails storage/document_packages

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["python", "main.py"]