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

# Copy source code and scripts early for preprocessing
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY data/ebm_sample.xml /app/data/ebm.xml

# Pre-build the vector store during image build to enable instant startup
RUN mkdir -p /app/data && \
    python scripts/download_full_ebm.py && \
    python scripts/build_database.py --xml /app/data/ebm.xml --store /app/data/vector_store

# Copy remaining files (Gradio app and configs)
COPY app.py /app/
COPY README.md /app/

EXPOSE 7860

CMD ["python", "app.py"]

