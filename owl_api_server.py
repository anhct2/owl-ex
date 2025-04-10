#!/usr/bin/env python
"""
Owl API Server
=============
Server API để expose dịch vụ Owl cho Manus-Owl project
Hỗ trợ cả xử lý đồng bộ và bất đồng bộ, cung cấp đầy đủ quá trình thực hiện
"""

import os
import sys
import json
import asyncio
import threading
import concurrent.futures
import queue
import re
import time
from fastapi import FastAPI, Body, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import uvicorn
import logging
import importlib
from typing import Dict, Any, Optional, List, Generator, AsyncGenerator

# Thiết lập logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("owl-api")

# Thêm đường dẫn hiện tại vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Dictionary chứa module descriptions (copy từ webapp.py)
MODULE_DESCRIPTIONS = {
    "run": "Default mode: Using OpenAI model's default agent collaboration mode, suitable for most tasks.",
    "run_mini": "Using OpenAI model with minimal configuration to process tasks",
    "run_gemini": "Using Gemini model to process tasks",
    "run_deepseek_zh": "Using deepseek model to process Chinese tasks",
    "run_openai_compatible_model": "Using openai compatible model to process tasks",
    "run_ollama": "Using local ollama model to process tasks",
    "run_qwen_mini_zh": "Using qwen model with minimal configuration to process tasks",
    "run_qwen_zh": "Using qwen model to process tasks",
    "run_azure_openai": "Using azure openai model to process tasks",
    "run_groq": "Using groq model to process tasks",
}

# Import run_society từ owl.utils
try:
    from owl.utils import run_society
    logger.info("Successfully imported run_society from owl.utils")
except ImportError as e:
    logger.error(f"Error importing run_society: {e}")
    raise ImportError(f"Could not import run_society from owl.utils: {e}")

# Direct import for Gemini API (for simple text-only queries)
try:
    from dotenv import load_dotenv
    load_dotenv()
    import google.generativeai as genai
    
    # Hardcoded key for testing (from the .env file)
    GEMINI_API_KEY = "AIzaSyBj94WPeYpo_0pqR-kXvhvLSLjxRr8dZQU"
    
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Successfully configured Gemini API for direct calls")
    else:
        logger.warning("GEMINI_API_KEY not found in environment variables")
except ImportError as e:
    logger.warning(f"Could not import Google Generative AI: {e}")
    logger.warning("Direct Gemini calls will not be available")

# Khởi tạo FastAPI
app = FastAPI(
    title="Owl API",
    description="API service for Owl multi-agent framework",
    version="1.0.0"
)

# Lưu trữ các task đang chạy và logs của chúng
ACTIVE_TASKS = {}  # task_id -> {logs: queue.Queue, events: list, status: str, result: dict}

class TaskRequest(BaseModel):
    title: str
    description: str
    options: Optional[Dict[str, Any]] = None

class TaskResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[float] = None

class StreamUpdate(BaseModel):
    type: str  # "log", "message", "status", "result"
    data: Any
    timestamp: float

# Function để validate input
def validate_input(question: str) -> bool:
    """Validate if user input is valid"""
    # Check if input is empty or contains only spaces
    if not question or question.strip() == "":
        return False
    return True

# Simple function to directly query Gemini for text-only prompts
def direct_gemini_query(prompt: str):
    """Query Gemini directly for simple text-only prompts"""
    if not GEMINI_API_KEY:
        return None, "GEMINI_API_KEY not configured in environment variables"
    
    try:
        logger.info(f"Making direct Gemini query: {prompt}")
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        if hasattr(response, 'text'):
            return response.text, None
        else:
            return str(response), None
    except Exception as e:
        logger.error(f"Error making direct Gemini query: {e}")
        return None, f"Error making direct Gemini query: {e}"

# Helper function để xử lý logs tương tự như trong webapp.py
def process_log(log: str) -> Optional[List[ChatMessage]]:
    """Process a log line to extract chat messages"""
    if "camel.agents.chat_agent - INFO" not in log:
        return None
    
    formatted_messages = []
    
    # Thử extract message array
    messages_match = re.search(
        r"Model (.*?), index (\d+), processed these messages: (\[.*\])", log
    )
    
    if messages_match:
        try:
            messages = json.loads(messages_match.group(3))
            for msg in messages:
                if msg.get("role") in ["user", "assistant"]:
                    content = msg.get("content", "").replace("\\n", "\n")
                    formatted_messages.append(
                        ChatMessage(
                            role=msg.get("role"),
                            content=content,
                            timestamp=time.time()
                        )
                    )
        except json.JSONDecodeError:
            pass
    
    # Nếu JSON parsing thất bại, thử extract conversation content trực tiếp
    if not formatted_messages:
        user_pattern = re.compile(r"\{'role': 'user', 'content': '(.*?)'\}")
        assistant_pattern = re.compile(
            r"\{'role': 'assistant', 'content': '(.*?)'\}"
        )
        
        for content in user_pattern.findall(log):
            content = content.replace("\\n", "\n")
            formatted_messages.append(
                ChatMessage(
                    role="user",
                    content=content,
                    timestamp=time.time()
                )
            )
        
        for content in assistant_pattern.findall(log):
            content = content.replace("\\n", "\n")
            formatted_messages.append(
                ChatMessage(
                    role="assistant",
                    content=content,
                    timestamp=time.time()
                )
            )
    
    return formatted_messages if formatted_messages else None

# Lớp LogReader để đọc log ghi bởi các agent
class LogReader:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.log_queue = queue.Queue()
        self.messages = []
        self.status = "initializing"
        self.result = None
        
        # Thêm vào active tasks
        ACTIVE_TASKS[task_id] = {
            "logs": self.log_queue,
            "events": [],
            "status": self.status,
            "result": None,
            "messages": self.messages
        }
    
    def add_log(self, log: str):
        """Add a log to the queue and process it"""
        self.log_queue.put(log)
        
        # Process log to extract messages
        extracted_messages = process_log(log)
        if extracted_messages:
            for msg in extracted_messages:
                event = StreamUpdate(
                    type="message",
                    data=msg.dict(),
                    timestamp=time.time()
                )
                self.messages.append(msg)
                ACTIVE_TASKS[self.task_id]["events"].append(event.dict())
                ACTIVE_TASKS[self.task_id]["messages"] = [m.dict() for m in self.messages]
    
    def update_status(self, status: str):
        """Update task status"""
        self.status = status
        ACTIVE_TASKS[self.task_id]["status"] = status
        event = StreamUpdate(
            type="status",
            data=status,
            timestamp=time.time()
        )
        ACTIVE_TASKS[self.task_id]["events"].append(event.dict())
    
    def set_result(self, result: Dict[str, Any]):
        """Set the final result"""
        self.result = result
        ACTIVE_TASKS[self.task_id]["result"] = result
        event = StreamUpdate(
            type="result",
            data=result,
            timestamp=time.time()
        )
        ACTIVE_TASKS[self.task_id]["events"].append(event.dict())
    
    def get_status(self) -> str:
        """Get current status"""
        return self.status
    
    def get_events(self) -> List[Dict[str, Any]]:
        """Get all events"""
        return ACTIVE_TASKS[self.task_id]["events"]
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all chat messages"""
        return ACTIVE_TASKS[self.task_id]["messages"]

# Custom logging handler để bắt log từ agent
class QueueHandler(logging.Handler):
    def __init__(self, log_reader: LogReader):
        super().__init__()
        self.log_reader = log_reader
    
    def emit(self, record):
        try:
            log_message = self.format(record)
            self.log_reader.add_log(log_message)
        except Exception as e:
            print(f"Error in QueueHandler.emit(): {e}")

# Function để run owl trong thread cho async process
def run_owl_in_thread(task_id: str, question: str, module_name: str = "run"):
    """Run Owl in a separate thread to avoid Playwright asyncio issues"""
    # Tạo log reader cho task này
    log_reader = LogReader(task_id)
    
    # Setup custom logging handler
    logger.info(f"Setting up logging for task {task_id}")
    queue_handler = QueueHandler(log_reader)
    queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger = logging.getLogger()
    root_logger.addHandler(queue_handler)
    
    try:
        log_reader.update_status("processing")
        logger.info(f"Processing question using module {module_name}: {question}")
        
        # For Gemini with simple text prompts, use direct API call
        if module_name == "run_gemini" and is_simple_text_prompt(question):
            logger.info("Using direct Gemini API call for simple text prompt")
            
            # Tạo message giả cho log
            user_msg = ChatMessage(
                role="user",
                content=question,
                timestamp=time.time()
            )
            log_reader.messages.append(user_msg)
            ACTIVE_TASKS[task_id]["messages"] = [m.dict() for m in log_reader.messages]
            event = StreamUpdate(
                type="message",
                data=user_msg.dict(),
                timestamp=time.time()
            )
            ACTIVE_TASKS[task_id]["events"].append(event.dict())
            
            text, error = direct_gemini_query(question)
            if error:
                log_reader.update_status("error")
                logger.error(f"Error with direct Gemini query: {error}")
                return None, None, error
            
            # Add the assistant message
            assistant_msg = ChatMessage(
                role="assistant",
                content=text,
                timestamp=time.time()
            )
            log_reader.messages.append(assistant_msg)
            ACTIVE_TASKS[task_id]["messages"] = [m.dict() for m in log_reader.messages]
            event = StreamUpdate(
                type="message",
                data=assistant_msg.dict(),
                timestamp=time.time()
            )
            ACTIVE_TASKS[task_id]["events"].append(event.dict())
            
            # Create a simplified token info structure
            token_info = {
                "completion_token_count": len(text.split()) if text else 0,
                "prompt_token_count": len(question.split())
            }
            
            log_reader.update_status("completed")
            return text, token_info, None
        
        # Dynamic import module
        try:
            logger.info(f"Importing module: examples.{module_name}")
            module = importlib.import_module(f"examples.{module_name}")
        except ImportError as ie:
            log_reader.update_status("error")
            logger.error(f"Unable to import module examples.{module_name}: {str(ie)}")
            return None, None, f"Unable to import module: examples.{module_name} - {str(ie)}"
        
        # Check if module has construct_society function
        if not hasattr(module, "construct_society"):
            log_reader.update_status("error")
            logger.error(f"construct_society function not found in module examples.{module_name}")
            return None, None, f"construct_society function not found in module examples.{module_name}"
        
        # Build society
        try:
            logger.info("Building society simulation...")
            society = module.construct_society(question)
        except Exception as e:
            log_reader.update_status("error")
            logger.error(f"Error building society: {str(e)}")
            return None, None, f"Error building society: {str(e)}"
        
        # Run society
        try:
            logger.info("Running society simulation...")
            answer, chat_history, token_info = run_society(society)
            logger.info("Society simulation completed successfully")
            
            log_reader.update_status("completed")
            return answer, token_info, None
        except Exception as e:
            log_reader.update_status("error")
            logger.error(f"Error running society: {str(e)}")
            return None, None, f"Error running society: {str(e)}"
    
    except Exception as e:
        log_reader.update_status("error")
        logger.error(f"Unexpected error: {str(e)}")
        return None, None, f"Unexpected error: {str(e)}"
    finally:
        # Cleanup: remove custom handler
        root_logger = logging.getLogger()
        root_logger.removeHandler(queue_handler)

# Helper to determine if a prompt is a simple text query
def is_simple_text_prompt(prompt: str) -> bool:
    """Check if the prompt is a simple text query that doesn't require browser/tools"""
    # Keywords that suggest complex tasks requiring tools
    complex_keywords = [
        "navigate", "search", "browse", "find", "amazon", 
        "google", "website", "open", "download", "visit",
        "write file", "save file", "create file", "read file",
        "execute", "run code", "screenshot", "image", "picture"
    ]
    
    # Check if prompt contains complex keywords
    lower_prompt = prompt.lower()
    for keyword in complex_keywords:
        if keyword in lower_prompt:
            return False
    
    # If no complex keywords found, assume it's a simple text prompt
    return True

# Stream events cho SSE endpoint
async def stream_events(task_id: str) -> AsyncGenerator[str, None]:
    """Stream events for Server-Sent Events"""
    if task_id not in ACTIVE_TASKS:
        yield json.dumps({"error": "Task not found"})
        return
    
    # Trả về tất cả events đã có
    for event in ACTIVE_TASKS[task_id]["events"]:
        yield json.dumps(event)
    
    # Check status to see if we're done
    if ACTIVE_TASKS[task_id]["status"] in ["completed", "error"]:
        return
    
    # Otherwise, keep polling for new events
    last_event_count = len(ACTIVE_TASKS[task_id]["events"])
    
    while ACTIVE_TASKS[task_id]["status"] not in ["completed", "error"]:
        current_count = len(ACTIVE_TASKS[task_id]["events"])
        
        # Send any new events
        if current_count > last_event_count:
            for i in range(last_event_count, current_count):
                yield json.dumps(ACTIVE_TASKS[task_id]["events"][i])
            
            last_event_count = current_count
        
        # Wait a bit
        await asyncio.sleep(0.5)
    
    # Send any final events
    current_count = len(ACTIVE_TASKS[task_id]["events"])
    if current_count > last_event_count:
        for i in range(last_event_count, current_count):
            yield json.dumps(ACTIVE_TASKS[task_id]["events"][i])

@app.post("/api/tasks", response_model=TaskResponse)
async def create_task(task_data: TaskRequest, background_tasks: BackgroundTasks):
    """
    Create and execute a new task using Owl multi-agent framework
    """
    try:
        logger.info(f"Processing task: {task_data.title}")
        logger.info(f"Task description: {task_data.description}")

        # Validate input
        if not validate_input(task_data.description):
            logger.warning("Invalid input")
            return {
                "success": False,
                "error": "Invalid input: description cannot be empty"
            }

        # Get module from options or use default
        module_name = "run"  # Default module
        if task_data.options and "module" in task_data.options:
            module_name = task_data.options["module"]
            logger.info(f"Using specified module: {module_name}")

        # Kiểm tra module có trong danh sách không
        if module_name not in MODULE_DESCRIPTIONS:
            logger.warning(f"Invalid module: {module_name}")
            return {
                "success": False,
                "error": f"Invalid module: {module_name}"
            }

        # Tạo task ID từ hash của mô tả
        task_id = f"owl_{abs(hash(task_data.description)) % 10000000000:010d}"
        
        # Start task in background
        background_tasks.add_task(
            lambda: concurrent.futures.ThreadPoolExecutor().submit(
                run_owl_in_thread, task_id, task_data.description, module_name
            )
        )
        
        # Trả về ID task để client có thể theo dõi progress
        return {
            "success": True,
            "data": {
                "owlTaskId": task_id,
                "status": "processing",
                "streamUrl": f"/api/tasks/{task_id}/stream"
            }
        }
    except Exception as e:
        logger.error(f"Error processing task: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """
    Get task status and result
    """
    if task_id not in ACTIVE_TASKS:
        return {
            "success": False,
            "error": "Task not found"
        }
    
    status = ACTIVE_TASKS[task_id]["status"]
    messages = ACTIVE_TASKS[task_id]["messages"]
    result = ACTIVE_TASKS[task_id]["result"]
    
    response_data = {
        "owlTaskId": task_id,
        "status": status,
        "messages": messages
    }
    
    if status == "completed" and result:
        response_data["result"] = result
    
    return {
        "success": True,
        "data": response_data
    }

@app.get("/api/tasks/{task_id}/stream")
async def stream_task(task_id: str, request: Request):
    """
    Stream task progress with Server-Sent Events
    """
    if task_id not in ACTIVE_TASKS:
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": "Task not found"}
        )
    
    return EventSourceResponse(stream_events(task_id))
    
@app.get("/api/tasks/{task_id}/messages")
async def get_task_messages(task_id: str):
    """
    Get all messages for a task
    """
    if task_id not in ACTIVE_TASKS:
        return {
            "success": False,
            "error": "Task not found"
        }
    
    messages = ACTIVE_TASKS[task_id]["messages"]
    
    return {
        "success": True,
        "data": {
            "owlTaskId": task_id,
            "messages": messages
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "ok", "service": "owl-api"}

@app.get("/")
async def root():
    """
    Root endpoint with service information
    """
    return {
        "name": "Owl API Service",
        "description": "API for Owl multi-agent framework with realtime streaming support",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/api/tasks", "method": "POST", "description": "Create and execute a task"},
            {"path": "/api/tasks/{task_id}", "method": "GET", "description": "Get task status and result"},
            {"path": "/api/tasks/{task_id}/stream", "method": "GET", "description": "Stream task progress with SSE"},
            {"path": "/api/tasks/{task_id}/messages", "method": "GET", "description": "Get all messages for a task"},
            {"path": "/health", "method": "GET", "description": "Health check endpoint"}
        ]
    }

if __name__ == "__main__":
    logger.info("Starting Owl API server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)