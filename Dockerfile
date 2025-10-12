FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# copy files
COPY . /app

# upgrade pip and install requirements
RUN pip install --upgrade pip
RUN pip install .

# Expose port
EXPOSE $PORT

# Default command
CMD uvicorn api:app --host 0.0.0.0 --port $PORT 
