#!/usr/bin/env python
# Kiểm tra và cài đặt thư viện sseclient-py trước tiên
import sys
import os

try:
    import sseclient
except ImportError:
    print("Installing required package: sseclient-py")
    os.system(f"{sys.executable} -m pip install sseclient-py")
    try:
        import sseclient
    except ImportError:
        print("Failed to install sseclient-py. Please install it manually: pip install sseclient-py")
        sys.exit(1)

# Nhập các thư viện khác sau khi đã đảm bảo sseclient đã được cài đặt
import requests
import json
import time
import threading
import queue

# API endpoint - using the owl_api_server.py
API_URL = "http://localhost:8000/api/tasks"

def test_gemini_api_streaming():
    """
    Test the Gemini model through the API endpoint with streaming updates
    """
    # Define the prompt - a simple text-only prompt that should use direct API
    prompt = "What is the Gemini AI model? Explain in a detailed way."
    
    # Prepare the request data with module specification
    payload = {
        "title": "Gemini API Test with Streaming",
        "description": prompt,
        "options": {
            "module": "run_gemini"  # Specify to use the Gemini module
        }
    }
    
    print(f"Sending API request to: {API_URL}")
    print(f"Using module: run_gemini")
    print(f"Prompt: {prompt}")
    
    try:
        # Send the POST request to the API
        response = requests.post(API_URL, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            
            if result.get("success", False):
                task_id = result["data"]["owlTaskId"]
                stream_url = result["data"]["streamUrl"]
                
                print(f"\nTask created with ID: {task_id}")
                print(f"Stream URL: {stream_url}")
                
                # Set up streaming
                print("\nStreaming updates in real-time:")
                print("-" * 50)
                
                # Start a thread to stream events
                stream_thread = threading.Thread(
                    target=stream_events, 
                    args=(f"http://localhost:8000{stream_url}",)
                )
                stream_thread.daemon = True
                stream_thread.start()
                
                # Poll for task status in the main thread
                poll_task_status(task_id)
                
                # Get final results
                get_task_result(task_id)
            else:
                print(f"Error from API: {result.get('error', 'Unknown error')}")
        else:
            print(f"HTTP Error: {response.status_code}")
            print(response.text)
    
    except requests.exceptions.ConnectionError:
        print(f"Connection Error: Could not connect to the API endpoint at {API_URL}")
        print("Make sure the owl_api_server.py is running and accessible.")
        print("Run: 'python owl_api_server.py' to start the server")
    except Exception as e:
        print(f"Error: {e}")

def stream_events(stream_url):
    """
    Stream events from the SSE endpoint
    """
    try:
        # Create a forever-streaming GET request
        stream_response = requests.get(stream_url, stream=True)
        client = sseclient.SSEClient(stream_response)
        
        # Create format placeholders for different types of events
        formats = {
            "message": "\n{role}: {content}",
            "status": "Status: {data}",
            "result": "\nFinal Result: {data}",
            "log": "Log: {data}"
        }
        
        # Process events
        for event in client.events():
            try:
                data = json.loads(event.data)
                event_type = data.get("type")
                
                if event_type == "message":
                    msg_data = data.get("data", {})
                    role = msg_data.get("role", "unknown")
                    content = msg_data.get("content", "")
                    
                    # Format and print message
                    if role == "user":
                        print(f"\n👤 User: {content}")
                    elif role == "assistant":
                        print(f"\n🤖 Assistant: {content}")
                    else:
                        print(f"\n{role}: {content}")
                
                elif event_type == "status":
                    status = data.get("data", "unknown")
                    print(f"\n📊 Status: {status}")
                    
                    # If completed or error, break the loop
                    if status in ["completed", "error"]:
                        break
                
                elif event_type == "result":
                    result_data = data.get("data", {})
                    if isinstance(result_data, dict) and "result" in result_data:
                        print(f"\n✅ Final Result: {result_data['result']}")
                    else:
                        print(f"\n✅ Result received")
                    
                elif event_type == "log":
                    log_data = data.get("data", "")
                    print(f"\n📝 Log: {log_data}")
            
            except json.JSONDecodeError:
                print(f"Error parsing event data: {event.data}")
            except Exception as e:
                print(f"Error processing event: {e}")
    
    except Exception as e:
        print(f"Error streaming events: {e}")

def poll_task_status(task_id):
    """
    Poll for task status until completed or error
    """
    status = "processing"
    max_polls = 60  # Max number of polls (60 * 2 seconds = 2 minutes max)
    polls = 0
    
    while status == "processing" and polls < max_polls:
        try:
            # Get task status
            response = requests.get(f"{API_URL}/{task_id}")
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("success", False):
                    status = result["data"]["status"]
                    
                    # If completed or error, break the loop
                    if status in ["completed", "error"]:
                        break
            
            # Wait before next poll
            time.sleep(2)
            polls += 1
        
        except Exception as e:
            print(f"Error polling task status: {e}")
            break

def get_task_result(task_id):
    """
    Get the final task result
    """
    try:
        response = requests.get(f"{API_URL}/{task_id}")
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("success", False):
                status = result["data"]["status"]
                
                print("\n" + "=" * 50)
                print(f"Task {task_id} - Final Status: {status}")
                
                if "result" in result["data"]:
                    print("\nResult:")
                    print("-" * 50)
                    print(result["data"]["result"])
                    print("-" * 50)
                
                # Print token counts if available
                if "tokenCount" in result["data"]:
                    print(f"Token count: {result['data']['tokenCount']}")
                
                print("=" * 50)
    
    except Exception as e:
        print(f"Error getting task result: {e}")

def test_complex_prompt():
    """
    Test a more complex prompt that requires tools/browser
    """
    # A complex prompt that would require browser
    prompt = "Navigate to Amazon.com and identify one product that is attractive to coders. Please provide me with the product name and price. No need to verify your answer."
    
    # Prepare the request data
    payload = {
        "title": "Complex Gemini Test",
        "description": prompt,
        "options": {
            "module": "run_gemini"
        }
    }
    
    print(f"\nTESTING COMPLEX PROMPT")
    print(f"Prompt: {prompt}")
    print(f"Note: This may need a few minutes to run as it involves browser navigation")
    
    try:
        # Send the request
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("success", False):
                task_id = result["data"]["owlTaskId"]
                stream_url = result["data"]["streamUrl"]
                
                print(f"\nTask created with ID: {task_id}")
                print(f"Stream URL: {stream_url}")
                
                # Set up streaming
                print("\nStreaming updates in real-time (this will show the browser navigation process):")
                print("-" * 50)
                
                # Start a thread to stream events
                stream_thread = threading.Thread(
                    target=stream_events, 
                    args=(f"http://localhost:8000{stream_url}",)
                )
                stream_thread.daemon = True
                stream_thread.start()
                
                # Poll for task status in the main thread
                poll_task_status(task_id)
                
                # Get final results
                get_task_result(task_id)
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"HTTP Error: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Run the test
    print("TESTING SIMPLE TEXT PROMPT WITH STREAMING...")
    test_gemini_api_streaming()
    
    # Test a complex prompt that requires browser/tools
    print("\nTESTING COMPLEX PROMPT (OPTIONAL)...")
    choice = input("Would you like to test a complex prompt that uses browser navigation? (y/n): ")
    if choice.lower() == 'y':
        test_complex_prompt()
    
    print("\nTest complete.")