#!/bin/bash

# Script để cài đặt và chạy Owl API server

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up Owl API server...${NC}"

# Cài đặt các dependencies
echo -e "\n${YELLOW}Installing required dependencies...${NC}"
pip install fastapi uvicorn pydantic

# Kiểm tra xem đã có môi trường .env chưa
if [ ! -f .env ]; then
    echo -e "\n${YELLOW}Creating .env file...${NC}"
    cp .env_template .env
    echo -e "${GREEN}Created .env file from template${NC}"
else
    echo -e "${GREEN}Found existing .env file${NC}"
fi

# Kiểm tra xem API keys đã được cấu hình chưa
echo -e "\n${YELLOW}Checking API keys configuration...${NC}"
grep -q "OPENAI_API_KEY" .env && echo -e "${GREEN}OpenAI API key found${NC}" || echo -e "${RED}OpenAI API key not found. You may need to add it to .env${NC}"

# Cấp quyền thực thi cho server script
echo -e "\n${YELLOW}Making API server executable...${NC}"
chmod +x owl_api_server.py

echo -e "\n${GREEN}Setup completed!${NC}"
echo -e "You can now run the Owl API server with: ${YELLOW}./owl_api_server.py${NC}"
echo -e "The server will be available at http://localhost:8000"
echo -e "Make sure your Manus-Owl backend is configured to connect to this URL"
echo -e "\nPress Enter to start the server now, or Ctrl+C to exit"
read

echo -e "\n${YELLOW}Starting Owl API server...${NC}"
./owl_api_server.py