FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev \
    libportmidi-dev \
    libswscale-dev \
    libavformat-dev \
    libavcodec-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --prefix=/install numpy pygame numba \
    torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html


FROM python:3.12-slim AS tester

WORKDIR /app

COPY --from=builder /app /app

COPY --from=builder /install/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

COPY . .

CMD ["python", "test.py"]
