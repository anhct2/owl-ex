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
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.messages import BaseMessage

from .progress_tracking import ScriptManager, Script, ScriptStep

logger = logging.getLogger(__name__)

class ScriptAutomation:
    """Manages script automation for OWL."""
    
    def __init__(
        self,
        script_manager: Optional[ScriptManager] = None,
        model=None
    ):
        """Initialize script automation.
        
        Args:
            script_manager: Manager for scripts
            model: LLM model to use for script generation
        """
        self.script_manager = script_manager or ScriptManager()
        self.model = model
        
        # Create default model if none provided
        if not self.model:
            try:
                self.model = ModelFactory.create(
                    model_platform=ModelPlatformType.OPENAI,
                    model_type=ModelType.GPT_4O,
                    model_config_dict={"temperature": 0},
                )
            except Exception as e:
                logger.warning(f"Failed to create default model: {str(e)}")
                self.model = None
    
    def create_script_from_task(self, task_description: str) -> str:
        """Create a script from a task description.
        
        Args:
            task_description: Description of the task
            
        Returns:
            str: ID of the created script
        """
        # Use the script manager to create a script
        task_id = self.script_manager.create_script(task_description)
        
        # If we have a model, enhance the script with more detailed steps
        if self.model:
            self._enhance_script_with_model(task_id, task_description)
            
        return task_id
    
    def _enhance_script_with_model(self, task_id: str, task_description: str) -> None:
        """Enhance a script with more detailed steps using an LLM.
        
        Args:
            task_id: ID of the script to enhance
            task_description: Description of the task
        """
        if not self.model:
            return
            
        try:
            # Get the current script
            script_data = self.script_manager.get_script(task_id)
            if not script_data:
                return
                
            # Create a prompt for the model
            prompt = f"""
            Task: {task_description}
            
            Please create a detailed step-by-step plan to accomplish this task. 
            For each step, provide:
            1. A clear description of what needs to be done
            2. The tools that might be needed (search, browser, terminal, file_write, etc.)
            3. The expected output of the step
            
            Format your response as a JSON array of steps, where each step has:
            - step_id: A unique identifier (e.g., "step_1", "step_2")
            - description: Detailed description of the step
            - tools: Array of tool names that might be needed
            - expected_output: What should be achieved after this step
            
            Example:
            [
                {{
                    "step_id": "step_1",
                    "description": "Research the latest information about X",
                    "tools": ["search", "browser"],
                    "expected_output": "Comprehensive understanding of X"
                }},
                {{
                    "step_id": "step_2",
                    "description": "Download and analyze data from Y",
                    "tools": ["browser", "terminal", "file_write"],
                    "expected_output": "Downloaded and processed data ready for analysis"
                }}
            ]
            """
            
            # Get response from the model
            response = self._get_model_response(prompt)
            
            # Parse the response to extract steps
            steps = self._parse_steps_from_response(response)
            
            if steps:
                # Update the script with the new steps
                script_data["steps"] = steps
                self.script_manager.update_script(task_id, script_data)
                
        except Exception as e:
            logger.error(f"Error enhancing script with model: {str(e)}")
    
    def _get_model_response(self, prompt: str) -> str:
        """Get a response from the model.
        
        Args:
            prompt: Prompt to send to the model
            
        Returns:
            str: Model response
        """
        try:
            # This is a simplified implementation
            # In a real implementation, this would use the appropriate API for the model
            if hasattr(self.model, "generate_response"):
                response = self.model.generate_response(prompt)
                return response.content
            elif hasattr(self.model, "generate"):
                response = self.model.generate(prompt)
                return response
            else:
                # Fallback to a simple placeholder
                return "[]"
        except Exception as e:
            logger.error(f"Error getting model response: {str(e)}")
            return "[]"
    
    def _parse_steps_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse steps from a model response.
        
        Args:
            response: Model response
            
        Returns:
            List[Dict]: Parsed steps
        """
        try:
            # Try to extract JSON from the response
            # First, look for JSON array
            import re
            json_match = re.search(r'\[\s*{.*}\s*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
                
            # If that fails, try to parse the entire response as JSON
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error parsing steps from response: {str(e)}")
            return []
    
    def update_script_progress(
        self, 
        task_id: str, 
        step_id: str, 
        status: str, 
        result: Optional[str] = None, 
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update the progress of a script step.
        
        Args:
            task_id: ID of the script
            step_id: ID of the step
            status: New status
            result: Result of the step execution
            error: Error message if the step failed
            
        Returns:
            Dict: Updated progress information
        """
        return self.script_manager.update_step_status(task_id, step_id, status, result, error)
    
    def get_next_step_prompt(self, task_id: str) -> Optional[str]:
        """Get a prompt for the next step in a script.
        
        Args:
            task_id: ID of the script
            
        Returns:
            str: Prompt for the next step or None if no more steps
        """
        current_step = self.script_manager.get_current_step(task_id)
        if not current_step:
            return None
            
        script_data = self.script_manager.get_script(task_id)
        if not script_data:
            return None
            
        # Create a prompt for the next step
        prompt = f"""
        Task: {script_data['task_description']}
        
        Current step: {current_step['description']}
        
        Please provide detailed instructions on how to accomplish this step.
        If specific tools are needed, explain how to use them effectively.
        
        Expected output: {current_step['expected_output']}
        """
        
        return prompt
    
    def get_script_progress(self, task_id: str) -> Dict[str, Any]:
        """Get the progress of a script.
        
        Args:
            task_id: ID of the script
            
        Returns:
            Dict: Progress information
        """
        return self.script_manager.progress_tracker.get_overall_progress(task_id)
    
    def get_script_report(self, task_id: str) -> Dict[str, Any]:
        """Get a detailed report for a script.
        
        Args:
            task_id: ID of the script
            
        Returns:
            Dict: Detailed report
        """
        return self.script_manager.progress_tracker.generate_progress_report(task_id)


class EnhancedOwlRolePlaying:
    """Enhances OwlRolePlaying with script automation capabilities."""
    
    def __init__(self, original_role_playing, script_automation=None):
        """Initialize enhanced role playing.
        
        Args:
            original_role_playing: Original OwlRolePlaying instance
            script_automation: Script automation instance
        """
        self.original = original_role_playing
        self.script_automation = script_automation or ScriptAutomation()
        self.task_id = None
        
    def init_with_task(self, task_description: str) -> str:
        """Initialize with a task description.
        
        Args:
            task_description: Description of the task
            
        Returns:
            str: ID of the created script
        """
        self.task_id = self.script_automation.create_script_from_task(task_description)
        return self.task_id
    
    def init_chat(self, init_msg_content: Optional[str] = None) -> BaseMessage:
        """Initialize chat with script-aware content.
        
        Args:
            init_msg_content: Initial message content
            
        Returns:
            BaseMessage: Initial message
        """
        # If we have a task ID, get the current step prompt
        if self.task_id:
            step_prompt = self.script_automation.get_next_step_prompt(self.task_id)
            if step_prompt and not init_msg_content:
                init_msg_content = step_prompt
                
            # Update the current step status to in_progress
            current_step = self.script_automation.script_manager.get_current_step(self.task_id)
            if current_step:
                self.script_automation.update_script_progress(
                    self.task_id,
                    current_step["step_id"],
                    "in_progress"
                )
        
        # Call the original init_chat
        return self.original.init_chat(init_msg_content)
    
    def step(self, assistant_msg: BaseMessage) -> Tuple[Any, Any]:
        """Perform a step in the conversation with script tracking.
        
        Args:
            assistant_msg: Message from the assistant
            
        Returns:
            Tuple: Assistant and user responses
        """
        # Call the original step method
        assistant_response, user_response = self.original.step(assistant_msg)
        
        # If we have a task ID, check if the step is completed
        if self.task_id and hasattr(user_response, "msg") and user_response.msg:
            user_content = user_response.msg.content
            
            # Check if the step is completed
            if "TASK_DONE" in user_content:
                # Get the current step
                current_step = self.script_automation.script_manager.get_current_step(self.task_id)
                if current_step:
                    # Update the step status to completed
                    self.script_automation.update_script_progress(
                        self.task_id,
                        current_step["step_id"],
                        "completed",
                        result=assistant_response.msg.content if hasattr(assistant_response, "msg") else None
                    )
                    
                    # Get the next step
                    next_step = self.script_automation.script_manager.move_to_next_step(self.task_id)
                    if next_step:
                        # Update the next step status to in_progress
                        self.script_automation.update_script_progress(
                            self.task_id,
                            next_step["step_id"],
                            "in_progress"
                        )
        
        return assistant_response, user_response
    
    async def astep(self, assistant_msg: BaseMessage) -> Tuple[Any, Any]:
        """Perform an async step in the conversation with script tracking.
        
        Args:
            assistant_msg: Message from the assistant
            
        Returns:
            Tuple: Assistant and user responses
        """
        # Call the original astep method
        assistant_response, user_response = await self.original.astep(assistant_msg)
        
        # If we have a task ID, check if the step is completed
        if self.task_id and hasattr(user_response, "msg") and user_response.msg:
            user_content = user_response.msg.content
            
            # Check if the step is completed
            if "TASK_DONE" in user_content or "任务已完成" in user_content:
                # Get the current step
                current_step = self.script_automation.script_manager.get_current_step(self.task_id)
                if current_step:
                    # Update the step status to completed
                    self.script_automation.update_script_progress(
                        self.task_id,
                        current_step["step_id"],
                        "completed",
                        result=assistant_response.msg.content if hasattr(assistant_response, "msg") else None
                    )
                    
                    # Get the next step
                    next_step = self.script_automation.script_manager.move_to_next_step(self.task_id)
                    if next_step:
                        # Update the next step status to in_progress
                        self.script_automation.update_script_progress(
                            self.task_id,
                            next_step["step_id"],
                            "in_progress"
                        )
        
        return assistant_response, user_response
    
    def __getattr__(self, name):
        """Delegate attribute access to the original instance.
        
        Args:
            name: Attribute name
            
        Returns:
            Any: Attribute value
        """
        return getattr(self.original, name)


def enhanced_run_society(
    society,
    round_limit: int = 15,
) -> Tuple[str, List[dict], dict]:
    """Enhanced version of run_society with script automation.
    
    Args:
        society: Society instance
        round_limit: Maximum number of rounds
        
    Returns:
        Tuple: Answer, chat history, token info
    """
    # Check if society is already enhanced
    if not hasattr(society, "script_automation"):
        # Enhance the society with script automation
        society = EnhancedOwlRolePlaying(society)
        
        # Initialize with the task description
        if hasattr(society.original, "task_prompt"):
            society.init_with_task(society.original.task_prompt)
    
    # Initialize tracking variables
    overall_completion_token_count = 0
    overall_prompt_token_count = 0
    chat_history = []
    
    # Initialize chat
    init_prompt = """
    Now please give me instructions to solve over overall task step by step. If the task requires some specific knowledge, please instruct me to use tools to complete the task.
    """
    input_msg = society.init_chat(init_prompt)
    
    # Main conversation loop
    for _round in range(round_limit):
        # Perform a step
        assistant_response, user_response = society.step(input_msg)
        
        # Track token usage
        if assistant_response.info.get("usage") and user_response.info.get("usage"):
            overall_completion_token_count += assistant_response.info["usage"].get(
                "completion_tokens", 0
            ) + user_response.info["usage"].get("completion_tokens", 0)
            overall_prompt_token_count += assistant_response.info["usage"].get(
                "prompt_tokens", 0
            ) + user_response.info["usage"].get("prompt_tokens", 0)
        
        # Process tool calls
        tool_call_records = []
        if assistant_response.info.get("tool_calls"):
            for tool_call in assistant_response.info["tool_calls"]:
                tool_call_records.append(tool_call.as_dict())
        
        # Update chat history
        _data = {
            "user": user_response.msg.content
            if hasattr(user_response, "msg") and user_response.msg
            else "",
            "assistant": assistant_response.msg.content
            if hasattr(assistant_response, "msg") and assistant_response.msg
            else "",
            "tool_calls": tool_call_records,
        }
        chat_history.append(_data)
        
        # Log the conversation
        logger.info(
            f"Round #{_round} user_response:\n {user_response.msgs[0].content if user_response.msgs and len(user_response.msgs) > 0 else ''}"
        )
        logger.info(
            f"Round #{_round} assistant_response:\n {assistant_response.msgs[0].content if assistant_response.msgs and len(assistant_response.msgs) > 0 else ''}"
        )
        
        # Check for termination conditions
        if (
            assistant_response.terminated
            or user_response.terminated
            or "TASK_DONE" in user_response.msg.content
        ):
            break
            
        input_msg = assistant_response.msg
    
    # Prepare results
    answer = chat_history[-1]["assistant"]
    token_info = {
        "completion_token_count": overall_completion_token_count,
        "prompt_token_count": overall_prompt_token_count,
    }
    
    # If society has script automation, include script progress in the result
    if hasattr(society, "script_automation") and society.task_id:
        token_info["script_progress"] = society.script_automation.get_script_progress(society.task_id)
    
    return answer, chat_history, token_info


async def enhanced_arun_society(
    society,
    round_limit: int = 15,
) -> Tuple[str, List[dict], dict]:
    """Enhanced async version of run_society with script automation.
    
    Args:
        society: Society instance
        round_limit: Maximum number of rounds
        
    Returns:
        Tuple: Answer, chat history, token info
    """
    # Check if society is already enhanced
    if not hasattr(society, "script_automation"):
        # Enhance the society with script automation
        society = EnhancedOwlRolePlaying(society)
        
        # Initialize with the task description
        if hasattr(society.original, "task_prompt"):
            society.init_with_task(society.original.task_prompt)
    
    # Initialize tracking variables
    overall_completion_token_count = 0
    overall_prompt_token_count = 0
    chat_history = []
    
    # Initialize chat
    init_prompt = """
    Now please give me instructions to solve over overall task step by step. If the task requires some specific knowledge, please instruct me to use tools to complete the task.
    """
    input_msg = society.init_chat(init_prompt)
    
    # Main conversation loop
    for _round in range(round_limit):
        # Perform a step
        assistant_response, user_response = await society.astep(input_msg)
        
        # Track token usage
        if assistant_response.info.get("usage") and user_response.info.get("usage"):
            overall_prompt_token_count += assistant_response.info["usage"].get(
                "completion_tokens", 0
            )
            overall_prompt_token_count += assistant_response.info["usage"].get(
                "prompt_tokens", 0
            ) + user_response.info["usage"].get("prompt_tokens", 0)
        
        # Process tool calls
        tool_call_records = []
        if assistant_response.info.get("tool_calls"):
            for tool_call in assistant_response.info["tool_calls"]:
                tool_call_records.append(tool_call.as_dict())
        
        # Update chat history
        _data = {
            "user": user_response.msg.content
            if hasattr(user_response, "msg") and user_response.msg
            else "",
            "assistant": assistant_response.msg.content
            if hasattr(assistant_response, "msg") and assistant_response.msg
            else "",
            "tool_calls": tool_call_records,
        }
        chat_history.append(_data)
        
        # Log the conversation
        logger.info(
            f"Round #{_round} user_response:\n {user_response.msgs[0].content if user_response.msgs and len(user_response.msgs) > 0 else ''}"
        )
        logger.info(
            f"Round #{_round} assistant_response:\n {assistant_response.msgs[0].content if assistant_response.msgs and len(assistant_response.msgs) > 0 else ''}"
        )
        
        # Check for termination conditions
        if (
            assistant_response.terminated
            or user_response.terminated
            or "TASK_DONE" in user_response.msg.content
            or "任务已完成" in user_response.msg.content
        ):
            break
            
        input_msg = assistant_response.msg
    
    # Prepare results
    answer = chat_history[-1]["assistant"]
    token_info = {
        "completion_token_count": overall_completion_token_count,
        "prompt_token_count": overall_prompt_token_count,
    }
    
    # If society has script automation, include script progress in the result
    if hasattr(society, "script_automation") and society.task_id:
        token_info["script_progress"] = society.script_automation.get_script_progress(society.task_id)
    
    return answer, chat_history, token_info
