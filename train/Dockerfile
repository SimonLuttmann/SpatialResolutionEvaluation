# Use an official Python runtime as a parent image
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /data

COPY yolov10/requirements.txt /data/requirements.txt

# Install Python dependencies from the copied requirements.txt
RUN pip install -r /data/requirements.txt

# Install the ultralytics package from PyPI
RUN pip install ultralytics

CMD ["bash"]
