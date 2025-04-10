#!/bin/bash

# Display banner
echo "======================================"
echo "   OWL API Server Docker Launcher    "
echo "======================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists, create it if it doesn't
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "# API Keys" > .env
    echo "OPENAI_API_KEY=" >> .env
    echo "GEMINI_API_KEY=" >> .env
    echo ".env file created. Please edit it to add your API keys."
fi

# Build and start the container
echo "Building and starting the Docker container..."
docker-compose up -d --build

# Check if the container is running
if [ $? -eq 0 ]; then
    echo ""
    echo "OWL API Server is now running!"
    echo "API is accessible at: http://localhost:8000"
    echo ""
    echo "To check logs, run: docker-compose logs -f"
    echo "To stop the server, run: docker-compose down"
else
    echo "Failed to start the container. Please check the logs."
    exit 1
fi 