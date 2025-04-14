#!/usr/bin/env python
# Kiểm tra và cài đặt thư viện sseclient-py trước tiên
import sys
import os
import argparse
import subprocess

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

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test OWL API with different models')
parser.add_argument('--api-url', default="http://localhost:9000", help='API URL (default: http://localhost:9000)')
args = parser.parse_args()

# API endpoint - using the owl_api_server.py
API_URL = f"{args.api_url}/api/tasks"
print(f"Using API URL: {API_URL}")

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
                    args=(f"{args.api_url}{stream_url}",)
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
    max_polls = 120  # Tăng lên 120 lần poll (120 * 2 seconds = 4 phút max)
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
            
            # Print a dot every 10 polls to show progress
            if polls % 10 == 0:
                print(".", end="", flush=True)
        
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
                    
                # Try to check if there's an answer field in case result is an object
                if isinstance(result["data"].get("result"), dict) and "answer" in result["data"]["result"]:
                    print("\nAnswer from result object:")
                    print("-" * 50)
                    print(result["data"]["result"]["answer"])
                    print("-" * 50)
                
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
                    args=(f"{args.api_url}{stream_url}",)
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

def test_recipe_prompt():
    """
    Test the recipe prompt from run_gemini.py default example
    """
    # The default recipe prompt from run_gemini.py
    prompt = "I have chicken breast, broccoli, garlic, and pasta. I'm looking for a quick dinner recipe that's healthy. I'm also trying to reduce my sodium intake. Search the internet for a recipe, modify it for low sodium, and create a shopping list for any additional ingredients I need?"
    
    # Prepare the request data
    payload = {
        "title": "Recipe Test with Gemini",
        "description": prompt,
        "options": {
            "module": "run_gemini"
        }
    }
    
    print(f"\nTESTING RECIPE PROMPT")
    print(f"Prompt: {prompt}")
    print(f"Note: This is the default example from run_gemini.py and may take several minutes to run")
    
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
                print("\nStreaming updates in real-time:")
                print("-" * 50)
                
                # Start a thread to stream events
                stream_thread = threading.Thread(
                    target=stream_events, 
                    args=(f"{args.api_url}{stream_url}",)
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
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

def test_github_stats_prompt():
    """
    Test a prompt to get GitHub stats and create a chart
    """
    # Advanced prompt that requires multiple tools: web browsing, code generation, file writing, and code execution
    prompt = "Open Google search, summarize the github stars, fork counts, etc. of camel-ai's camel framework, and write the numbers into a python file using the plot package, save it locally, and run the generated python file."
    
    # Prepare the request data with module specification
    payload = {
        "title": "GitHub Stats Chart Generation",
        "description": prompt,
        "options": {
            "module": "run_gemini"  # Using Gemini for this complex task
        }
    }
    
    print(f"\nTESTING GITHUB STATS VISUALIZATION PROMPT")
    print(f"Prompt: {prompt}")
    print(f"Note: This is a complex test that involves web browsing, code generation, and execution")
    
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
                    args=(f"{args.api_url}{stream_url}",)
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
    except Exception as e:
        print(f"Error: {e}")

def test_deepseek_prompt():
    """
    Test the DeepSeek model with a complex prompt
    """
    # Complex prompt for DeepSeek
    prompt = "Search for information about quantum computing advancements in 2024, summarize the main breakthroughs, and create a timeline of key events. Format the results as a markdown report."
    
    # Prepare the request data
    payload = {
        "title": "DeepSeek Quantum Computing Research",
        "description": prompt,
        "options": {
            "module": "run_deepseek"  # Using our new DeepSeek module
        }
    }
    
    print(f"\nTESTING DEEPSEEK MODEL")
    print(f"Prompt: {prompt}")
    print(f"Note: This test uses the DeepSeek model for complex research and summary")
    
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
                print("\nStreaming updates in real-time:")
                print("-" * 50)
                
                # Start a thread to stream events
                stream_thread = threading.Thread(
                    target=stream_events, 
                    args=(f"{args.api_url}{stream_url}",)
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
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

def check_api_status(api_url, timeout=5):
    """
    Check if the API server is running and responding.
    Returns: 
        (bool, str) - (success, message)
    """
    try:
        health_response = requests.get(f"{api_url}/health", timeout=timeout)
        if health_response.status_code == 200:
            return True, "✅ API server is running and responding!"
        else:
            return False, f"⚠️ API server responded with status code: {health_response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "❌ Could not connect to API server!"
    except Exception as e:
        return False, f"⚠️ Error checking API connection: {e}"

def start_api_server():
    """
    Attempt to start the API server.
    """
    try:
        print("Attempting to start API server...")
        subprocess.Popen(["python", "owl_api_server.py"], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE, 
                         start_new_session=True)
        
        # Give it a few seconds to start
        time.sleep(5)
        
        return "API server process started. Checking status..."
    except Exception as e:
        return f"Error starting API server: {e}"

if __name__ == "__main__":
    # Hiển thị thông tin hướng dẫn
    print("=" * 70)
    print("OWL API TEST - MULTI-MODEL BROWSER TESTS")
    print("=" * 70)
    print(f"API URL: {API_URL}")
    print("BEFORE RUNNING TESTS:")
    print("1. Make sure owl_api_server.py is running with the command:")
    print("   python owl_api_server.py")
    print("2. Or use the start_servers.sh script:")
    print("   ./start_servers.sh")
    print("=" * 70)
    
    # Kiểm tra kết nối API trước khi bắt đầu
    api_ready, status_message = check_api_status(args.api_url)
    print(status_message)
    
    if not api_ready:
        print(f"Make sure owl_api_server.py is running at {args.api_url}")
        choice = input("Do you want to try to start the API server? (y/n): ")
        
        if choice.lower() == 'y':
            result = start_api_server()
            print(result)
            
            # Check again
            api_ready, status_message = check_api_status(args.api_url)
            print(status_message)
            
            if not api_ready:
                choice = input("API server still not responding. Continue anyway? (y/n): ")
                if choice.lower() != 'y':
                    print("Exiting. Please start the API server manually and try again.")
                    sys.exit(1)
        else:
            choice = input("Continue anyway? (y/n): ")
            if choice.lower() != 'y':
                print("Exiting. Please start the API server and try again.")
                sys.exit(1)
    
    print("\n")
    
    # Run the test
    print("TESTING SIMPLE TEXT PROMPT WITH STREAMING...")
    test_gemini_api_streaming()
    
    # Test options for more complex prompts
    print("\nSelect a test to run:")
    print("1. Complex prompt with browser navigation")
    print("2. Recipe prompt from run_gemini.py default example")
    print("3. GitHub stats visualization prompt")
    print("4. DeepSeek quantum computing research")
    print("5. Run all tests")
    print("6. Skip additional tests")
    
    choice = input("Enter your choice (1-6): ")
    
    if choice == "1":
        test_complex_prompt()
    elif choice == "2":
        test_recipe_prompt()
    elif choice == "3":
        test_github_stats_prompt()
    elif choice == "4":
        test_deepseek_prompt()
    elif choice == "5":
        test_complex_prompt()
        test_recipe_prompt()
        test_github_stats_prompt()
        test_deepseek_prompt()
    else:
        print("Skipping additional tests.")
    
    print("\nTest complete.")