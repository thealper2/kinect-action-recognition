# Training / offline evaluation image. Kinect USB access is NOT supported inside Docker;
# live inference uses the Windows bridge (docs/kinect_setup.md).
#
#   docker build -t kard-skeleton-action .
#   docker run --gpus all --rm -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" \
#       kard-skeleton-action make train-stgcn
#
# The raw dataset is mounted, never baked into the image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 VENV=/usr/local
RUN apt-get update && apt-get install -y --no-install-recommends make && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CUDA 12.8 wheels (required for RTX 50xx); use --index-url .../cpu for a CPU-only image.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt requirements-dev.txt ./
RUN grep -v '^opencv' requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt pytest ruff nbformat nbclient ipykernel \
       jupyter nbconvert opencv-python-headless

COPY pyproject.toml README.md Makefile ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests
COPY notebooks ./notebooks
RUN pip install --no-cache-dir --no-deps -e .

VOLUME ["/app/data", "/app/outputs"]
CMD ["make", "help"]
