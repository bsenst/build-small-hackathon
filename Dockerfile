FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all source code, scripts, and data needed for runtime initialization
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY data/ /app/data/
COPY app.py /app/
COPY README.md /app/

EXPOSE 7860

CMD ["python", "app.py"]

