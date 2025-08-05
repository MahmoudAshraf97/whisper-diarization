FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    cython3 \
    build-essential \
    cmake \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install pybind11
RUN pip install whisperX
RUN pip install "pyarrow==20.0.0" "datasets==2.14.4"
RUN pip install python-dotenv

COPY requirements.txt constraints.txt ./
RUN pip install --upgrade pip && \
    pip install -c constraints.txt -r requirements.txt 


COPY . .

ENTRYPOINT ["python", "diarize.py"]