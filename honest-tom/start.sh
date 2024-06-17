#!/bin/bash

# Change these as you see fit
IMAGE_NAME="flask-api"
CONTAINER_NAME="flask-api"
PORT=5000

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Run the Docker container
echo "Running Docker container..."
docker run -d --name $CONTAINER_NAME -p $PORT:$PORT $IMAGE_NAME