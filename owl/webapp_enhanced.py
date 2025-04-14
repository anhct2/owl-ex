# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

import os
import gradio as gr
import time
import json
import logging
import datetime
from typing import Tuple, Dict, Any, List, Optional
import importlib
from dotenv import load_dotenv, set_key, find_dotenv, unset_key
import threading
import queue
import re

# Import from the correct module path
from .utils import (
    run_society, 
    enhanced_run_society,
    ScriptManager,
    ScriptAutomation,
    SandboxManager
)

os.environ["PYTHONIOENCODING"] = "utf-8"

# Configure logging system
def setup_logging():
    """Configure logging system to output logs to file, memory queue, and console"""
    # Create logs directory (if it doesn't exist)
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    # Generate log filename (using current date)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(logs_dir, f"gradio_log_{current_date}.txt")
    # Configure root logger (captures all logs)
    root_logger = logging.getLogger()
    # Clear existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.INFO)
    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    file_handler.setLevel(logging.INFO)
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    logging.info("Logging system initialized, log file: %s", log_file)
    return log_file

# Global variables
LOG_FILE = None
LOG_QUEUE: queue.Queue = queue.Queue()  # Log queue
STOP_LOG_THREAD = threading.Event()
CURRENT_PROCESS = None  # Used to track the currently running process
STOP_REQUESTED = threading.Event()  # Used to mark if stop was requested

# Script and Sandbox managers
SCRIPT_MANAGER = ScriptManager()
SANDBOX_MANAGER = SandboxManager()

# Log reading and updating functions
def log_reader_thread(log_file):
    """Background thread that continuously reads the log file and adds new lines to the queue"""
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            # Move to the end of file
            f.seek(0, 2)
            while not STOP_LOG_THREAD.is_set():
                line = f.readline()
                if line:
                    LOG_QUEUE.put(line)  # Add to conversation record queue
                else:
                    # No new lines, wait for a short time
                    time.sleep(0.1)
    except Exception as e:
        logging.error(f"Log reader thread error: {str(e)}")

def update_conversation_from_logs(conversation_box):
    """Update the conversation box with new log entries"""
    try:
        # Get all available log entries
        logs = []
        while not LOG_QUEUE.empty():
            logs.append(LOG_QUEUE.get_nowait())

        if logs:
            # Process and add to conversation
            conversation_text = conversation_box or ""
            for log in logs:
                # Only add user and assistant responses to the conversation
                if "user_response" in log or "assistant_response" in log:
                    # Clean up the log entry
                    log = re.sub(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - \w+ - \w+ - ", "", log)
                    log = log.strip()
                    conversation_text += log + "\n\n"
            return conversation_text
        return conversation_box
    except Exception as e:
        logging.error(f"Error updating conversation: {str(e)}")
        return conversation_box

def update_script_progress(task_id):
    """Update the script progress display"""
    if not task_id:
        return None
        
    try:
        progress = SCRIPT_MANAGER.progress_tracker.get_overall_progress(task_id)
        return json.dumps(progress, indent=2)
    except Exception as e:
        logging.error(f"Error updating script progress: {str(e)}")
        return json.dumps({"error": str(e)})

def update_sandbox_status(sandbox_id):
    """Update the sandbox status display"""
    if not sandbox_id:
        return None
        
    try:
        status = SANDBOX_MANAGER.get_sandbox_status(sandbox_id)
        return json.dumps(status, indent=2)
    except Exception as e:
        logging.error(f"Error updating sandbox status: {str(e)}")
        return json.dumps({"error": str(e)})

# Input validation
def validate_input(text):
    """Validate user input to prevent injection attacks"""
    if not text or not isinstance(text, str):
        return False
    # Check for reasonable length
    if len(text) > 10000:
        return False
    return True

# Module descriptions for UI
MODULE_DESCRIPTIONS = {
    "run_terminal": "Standard terminal-based agent with search, browser, file, and terminal tools",
    "run_terminal_sandbox": "Enhanced terminal-based agent with sandbox integration",
    "run_terminal_zh": "Chinese language terminal-based agent",
    "run_mcp": "Multi-agent collaboration protocol",
    "run_mcp_sse": "Multi-agent collaboration with server-sent events",
    "run_gemini": "Google Gemini model integration",
    "run_groq": "Groq model integration",
    "run_mistral": "Mistral model integration",
    "run_ollama": "Ollama model integration",
    "run_qwen_zh": "Alibaba Qwen model (Chinese)",
    "run_together_ai": "Together.ai model integration",
}

def run_owl(question: str, example_module: str) -> Tuple[str, str, str]:
    """Run the OWL system and return results
    Args:
        question: User question
        example_module: Example module name to import (e.g., "run_terminal_zh" or "run_deep")
    Returns:
        Tuple[...]: Answer, token count, status
    """
    global CURRENT_PROCESS
    # Validate input
    if not validate_input(question):
        logging.warning("User submitted invalid input")
        return (
            "Please enter a valid question",
            "0",
            "❌ Error: Invalid input question",
        )
    try:
        # Ensure environment variables are loaded
        load_dotenv(find_dotenv(), override=True)
        logging.info(
            f"Processing question: '{question}', using module: {example_module}"
        )
        # Check if the module is in MODULE_DESCRIPTIONS
        if example_module not in MODULE_DESCRIPTIONS:
            logging.error(f"User selected an unsupported module: {example_module}")
            return (
                f"Selected module '{example_module}' is not supported",
                "0",
                "❌ Error: Unsupported module",
            )
        # Dynamically import target module
        module_path = f"examples.{example_module}"
        try:
            logging.info(f"Importing module: {module_path}")
            module = importlib.import_module(module_path)
        except ImportError as ie:
            logging.error(f"Unable to import module {module_path}: {str(ie)}")
            return (
                f"Unable to import module: {module_path}",
                "0",
                f"❌ Error: Module {example_module} does not exist or cannot be loaded - {str(ie)}",
            )
        except Exception as e:
            logging.error(
                f"Error occurred while importing module {module_path}: {str(e)}"
            )
            return (
                f"Error occurred while importing module: {module_path}",
                "0",
                f"❌ Error: {str(e)}",
            )
        # Check if it contains the construct_society function
        if not hasattr(module, "construct_society"):
            logging.error(
                f"construct_society function not found in module {module_path}"
            )
            return (
                f"construct_society function not found in module {module_path}",
                "0",
                "❌ Error: Module interface incompatible",
            )
        # Build society simulation
        try:
            logging.info("Building society simulation...")
            society = module.construct_society(question)
        except Exception as e:
            logging.error(f"Error occurred while building society simulation: {str(e)}")
            return (
                f"Error occurred while building society simulation: {str(e)}",
                "0",
                f"❌ Error: Build failed - {str(e)}",
            )
        # Run society simulation
        try:
            logging.info("Running society simulation...")
            
            # Check if we should use enhanced_run_society
            if hasattr(society, "script_automation"):
                answer, chat_history, token_info = enhanced_run_society(society)
            else:
                answer, chat_history, token_info = run_society(society)
                
            logging.info("Society simulation completed")
        except Exception as e:
            logging.error(f"Error occurred while running society simulation: {str(e)}")
            return (
                f"Error occurred while running society simulation: {str(e)}",
                "0",
                f"❌ Error: Run failed - {str(e)}",
            )
        # Safely get token count
        if not isinstance(token_info, dict):
            token_info = {}
        completion_tokens = token_info.get("completion_token_count", 0)
        prompt_tokens = token_info.get("prompt_token_count", 0)
        total_tokens = completion_tokens + prompt_tokens
        
        # Get script progress if available
        script_progress = None
        if "script_progress" in token_info:
            script_progress = token_info["script_progress"]
            
        logging.info(
            f"Processing completed, token usage: completion={completion_tokens}, prompt={prompt_tokens}, total={total_tokens}"
        )
        
        # Return with script progress if available
        if script_progress:
            return (
                answer,
                f"Completion tokens: {completion_tokens:,} | Prompt tokens: {prompt_tokens:,} | Total: {total_tokens:,}",
                f"✅ Successfully completed with script progress: {json.dumps(script_progress, indent=2)}",
            )
        else:
            return (
                answer,
                f"Completion tokens: {completion_tokens:,} | Prompt tokens: {prompt_tokens:,} | Total: {total_tokens:,}",
                "✅ Successfully completed",
            )
    except Exception as e:
        logging.error(
            f"Uncaught error occurred while processing the question: {str(e)}"
        )
        return (f"Error occurred: {str(e)}", "0", f"❌ Error: {str(e)}")

def update_module_description(module_name: str) -> str:
    """Return the description of the selected module"""
    return MODULE_DESCRIPTIONS.get(module_name, "No description available")

# Store environment variables configured from the frontend
WEB_FRONTEND_ENV_VARS: dict[str, str] = {}

def init_env_file():
    """Initialize .env file if it doesn't exist"""
    dotenv_path = find_dotenv()
    if not dotenv_path:
        with open(".env", "w") as f:
            f.write("# OWL Environment Variables\n")
        dotenv_path = find_dotenv()
    return dotenv_path

def create_ui():
    """Create the Gradio UI"""
    # Initialize environment
    init_env_file()
    
    # Create UI blocks
    with gr.Blocks(
        title="OWL - Open-source Workflow Learning Agent",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .status-box { min-height: 50px; }
        .token-box { min-height: 30px; }
        .conversation-box { min-height: 400px; }
        """
    ) as demo:
        gr.Markdown(
            """
            # 🦉 OWL - Open-source Workflow Learning Agent
            
            OWL is an open-source workflow learning agent that can help you with various tasks.
            
            Enter your question below and click "Run" to start the conversation.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Enter your question here...",
                    lines=3
                )
                
                with gr.Row():
                    module_dropdown = gr.Dropdown(
                        choices=list(MODULE_DESCRIPTIONS.keys()),
                        value="run_terminal_sandbox",
                        label="Agent Type"
                    )
                    run_button = gr.Button("Run", variant="primary")
                    stop_button = gr.Button("Stop", variant="stop")
                    
                module_description = gr.Markdown(
                    update_module_description("run_terminal_sandbox")
                )
                
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Status"):
                        status_box = gr.Markdown(
                            "Ready to process your question.",
                            elem_classes=["status-box"]
                        )
                        token_box = gr.Markdown(
                            "Token usage will appear here.",
                            elem_classes=["token-box"]
                        )
                    
                    with gr.Tab("Script"):
                        task_id_input = gr.Textbox(
                            label="Task ID",
                            placeholder="Enter task ID to view progress",
                            visible=True
                        )
                        script_progress = gr.JSON(
                            label="Script Progress",
                            value=None
                        )
                        update_script_btn = gr.Button("Update Script Progress")
                    
                    with gr.Tab("Sandbox"):
                        sandbox_id_input = gr.Textbox(
                            label="Sandbox ID",
                            placeholder="Enter sandbox ID to view status",
                            visible=True
                        )
                        sandbox_status = gr.JSON(
                            label="Sandbox Status",
                            value=None
                        )
                        update_sandbox_btn = gr.Button("Update Sandbox Status")
        
        conversation_box = gr.Textbox(
            label="Conversation",
            placeholder="Conversation will appear here...",
            lines=20,
            max_lines=50,
            elem_classes=["conversation-box"]
        )
        
        # Set up event handlers
        def on_run_click(question, module_name):
            global CURRENT_PROCESS, STOP_REQUESTED
            STOP_REQUESTED.clear()
            
            # Start log reader thread if not already running
            if LOG_FILE and not CURRENT_PROCESS:
                STOP_LOG_THREAD.clear()
                CURRENT_PROCESS = threading.Thread(
                    target=log_reader_thread, args=(LOG_FILE,)
                )
                CURRENT_PROCESS.daemon = True
                CURRENT_PROCESS.start()
                
            # Run OWL
            answer, token_info, status = run_owl(question, module_name)
            
            # Extract task_id and sandbox_id if available
            task_id = None
            sandbox_id = None
            
            if "script progress" in status:
                # Try to extract task_id from the status
                try:
                    progress_json = json.loads(status.split("script progress: ")[1])
                    if "task_id" in progress_json:
                        task_id = progress_json["task_id"]
                except Exception:
                    pass
            
            # Update UI
            return answer, token_info, status, task_id, sandbox_id
        
        def on_stop_click():
            global STOP_REQUESTED
            STOP_REQUESTED.set()
            return "Processing stopped by user.", "0", "⚠️ Stopped by user."
        
        def on_module_change(module_name):
            return update_module_description(module_name)
        
        # Connect event handlers
        run_button.click(
            fn=on_run_click,
            inputs=[question_input, module_dropdown],
            outputs=[conversation_box, token_box, status_box, task_id_input, sandbox_id_input]
        )
        
        stop_button.click(
            fn=on_stop_click,
            inputs=[],
            outputs=[conversation_box, token_box, status_box]
        )
        
        module_dropdown.change(
            fn=on_module_change,
            inputs=[module_dropdown],
            outputs=[module_description]
        )
        
        # Script progress update
        update_script_btn.click(
            fn=update_script_progress,
            inputs=[task_id_input],
            outputs=[script_progress]
        )
        
        # Sandbox status update
        update_sandbox_btn.click(
            fn=update_sandbox_status,
            inputs=[sandbox_id_input],
            outputs=[sandbox_status]
        )
        
        # Periodic updates
        demo.load(lambda: None)
        
        # Periodic updates
        demo.load(
            update_conversation_from_logs,
            inputs=conversation_box,
            outputs=conversation_box,
            # every=1
        )
        
    return demo

if __name__ == "__main__":
    # Set up logging
    LOG_FILE = setup_logging()
    
    # Create and launch UI
    demo = create_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=False)
