FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app
VOLUME /app
# install system deps
RUN apt-get update && apt-get install -y build-essential
RUN apt-get update && apt-get install -y libgl1 libglx-mesa0 libglib2.0-0
RUN rm -rf /var/lib/apt/lists/*

# copy files
COPY . /app

# upgrade pip and install requirements
RUN pip install --upgrade pip
RUN pip install .

EXPOSE $PORT
 
CMD uvicorn api:app --host 0.0.0.0 --port $PORT 
