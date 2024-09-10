# Bachelor Thesis Repository

Welcome to the repository for my Bachelor Thesis, which focuses on training and analyzing a pre-trained YOLOv10x model.

## Prerequisites

Before you begin, ensure you have installed Docker.

## Repository Structure

This repository is organized as follows:
- `/train`: Contains the Dockerfile and a shell script and the python script for starting the training
- `/evaluate`: Includes scripts for model evaluation. Also jsons from my own evaulation are included.

## Setup and Deployment
- For the training utilize the shell script for easy Docker deployment and then start the train.py script inside the Docker Container.

- For the evaluation of performance on an NVIDIA Jetson Orin NX with Jetpack 5.x installed utilize the following command and then run the python scripts inside the Docker Container!

t=ultralytics/ultralytics:latest-jetson-jetpack5 && sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia $t
