FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Task 1(a): the GPU verification script runs on container start.
# Task 2 Jobs override this command in their YAML, so one image serves both tasks.
CMD ["python3", "src/container_nautilus_gpu_test.py"]
