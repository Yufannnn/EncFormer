FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LD_LIBRARY_PATH=/app/third_party/phantom-fhe/build/lib:/usr/local/cuda/lib64
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake libgmp-dev libssl-dev libomp-dev python3 python3-pip && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 && pip install --no-cache-dir -r /tmp/requirements.txt
COPY . .
RUN bash scripts/build.sh 70 && python scripts/check.py
ENTRYPOINT ["python", "scripts/demo.py"]
CMD ["--idx", "40", "--gpu", "0"]
