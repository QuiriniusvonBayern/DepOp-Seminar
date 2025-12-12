# ============================================
# Dockerfile for "Word Embedding for Semantic SQL in RDBMS"
# ============================================

FROM python:3.11-slim

LABEL maintainer="Lasse Wriedt <lasse.wriedt@uni-ulm.de>"
LABEL description="Docker container for Semantic SQL with Word Embeddings"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget build-essential \
    libblas-dev liblapack-dev libopenblas-dev gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip &&\
    pip install --no-cache-dir -r requirements.txt 

RUN python -m spacy download en_core_web_sm

RUN useradd -ms /bin/bash user
USER user

COPY --chown=user:user . .

# Declare persistent volume mount point
VOLUME ["/workspace"]

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--notebook-dir=/workspace"]