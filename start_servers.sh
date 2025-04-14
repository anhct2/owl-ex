#!/bin/bash

echo "Starting Owl API Server..."
python owl_api_server.py &
API_PID=$!

echo "Starting Frontend Server..."
python server.py &
FRONTEND_PID=$!

echo "Both servers are running!"
echo "API Server: http://localhost:9000"
echo "Frontend: http://localhost:8080"
echo "Press Ctrl+C to stop both servers"

# Handle termination
trap "kill $API_PID $FRONTEND_PID; exit" INT TERM
wait 