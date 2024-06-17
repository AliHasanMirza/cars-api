#!/bin/bash

IMAGE_NAME="carprice"

# Build the image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Launch the image
echo "Launching Docker container..."
docker run -it --rm -p 8888:8888 -v $(pwd):/home/jovyan/work $IMAGE_NAME
