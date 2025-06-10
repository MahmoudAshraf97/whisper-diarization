FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    cython3 \
    build-essential \
    cmake \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install pybind11 globally for C++ extensions
RUN pip install pybind11

# Copy requirement and constraint files
COPY requirements.txt constraints.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -c constraints.txt -r requirements.txt

# Copy your entire project code
COPY . .

# Set default run command
ENTRYPOINT ["python", "diarize.py"]