# Owl API Server - Docker Deployment Guide

This guide explains how to deploy the Owl API server using Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- API keys for services like OpenAI, Gemini, etc.

## Quick Start

### 1. Setup Environment Variables

Kiểm tra file `.env` trong thư mục gốc đã có sẵn. File này sẽ được sao chép vào container trong quá trình build.

```
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
# Add other API keys as needed
```

### 2. Build and Run with Docker Compose

```bash
# Build and start the service
docker-compose up -d

# Check logs
docker-compose logs -f
```

The API will be accessible at: `http://localhost:8000`

### 3. API Endpoints

- `GET /`: Root endpoint with service information
- `POST /api/tasks`: Create and execute a task
- `GET /api/tasks/{task_id}`: Get task status and result
- `GET /api/tasks/{task_id}/stream`: Stream task progress with SSE
- `GET /api/tasks/{task_id}/messages`: Get all messages for a task
- `GET /health`: Health check endpoint

## Docker Commands

```bash
# Build the image
docker-compose build

# Start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

## Environment Variables

Các biến môi trường được định nghĩa trong file `.env`:

- `OPENAI_API_KEY`: OpenAI API key
- `GEMINI_API_KEY`: Gemini API key
- Các API key khác nếu cần

**Lưu ý:** Nếu bạn cần thay đổi API key sau khi đã build container, bạn cần sửa file `.env` và build lại container:
```bash
docker-compose up -d --build
```

## Advanced Configuration

### Modifying the Dockerfile

If you need to install additional dependencies, edit the `Dockerfile`:

```dockerfile
# Add additional dependencies
RUN pip install --no-cache-dir additional-package
```

### Customizing Docker Compose

To modify port, volumes, or environment variables, edit `docker-compose.yml`:

```yaml
services:
  owl-api:
    ports:
      - "8888:8000"  # Change host port from 8000 to 8888
    volumes:
      - ./custom-data:/app/data  # Mount additional volumes
```

## Troubleshooting

### Container fails to start

Check logs for errors:

```bash
docker-compose logs
```

### API key issues

Kiểm tra file `.env` đã có API key chưa và xác nhận container đã được build lại sau khi thay đổi. 