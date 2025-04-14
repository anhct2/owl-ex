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
import time
import uuid
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class ScriptStep:
    """Represents a single step in a script execution plan."""
    
    def __init__(
        self,
        step_id: str,
        description: str,
        tools: Optional[List[str]] = None,
        expected_output: Optional[str] = None,
    ):
        """Initialize a script step.
        
        Args:
            step_id: Unique identifier for the step
            description: Description of what the step does
            tools: List of tools required for this step
            expected_output: Expected output description
        """
        self.step_id = step_id
        self.description = description
        self.tools = tools or []
        self.expected_output = expected_output
        self.status = "pending"  # pending, in_progress, completed, failed
        self.start_time = None
        self.end_time = None
        self.result = None
        self.error = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary representation."""
        return {
            "step_id": self.step_id,
            "description": self.description,
            "tools": self.tools,
            "expected_output": self.expected_output,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "result": self.result,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScriptStep':
        """Create a step from dictionary representation."""
        step = cls(
            step_id=data["step_id"],
            description=data["description"],
            tools=data.get("tools", []),
            expected_output=data.get("expected_output")
        )
        step.status = data.get("status", "pending")
        step.start_time = data.get("start_time")
        step.end_time = data.get("end_time")
        step.result = data.get("result")
        step.error = data.get("error")
        return step


class Script:
    """Represents a complete execution script with multiple steps."""
    
    def __init__(
        self,
        task_id: str,
        task_description: str,
        steps: Optional[List[ScriptStep]] = None
    ):
        """Initialize a script.
        
        Args:
            task_id: Unique identifier for the task
            task_description: Description of the overall task
            steps: List of script steps
        """
        self.task_id = task_id
        self.task_description = task_description
        self.steps = steps or []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.current_step_index = 0
        
    def add_step(self, step: ScriptStep) -> None:
        """Add a step to the script."""
        self.steps.append(step)
        self.updated_at = datetime.now().isoformat()
        
    def get_current_step(self) -> Optional[ScriptStep]:
        """Get the current step in the script."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    def move_to_next_step(self) -> Optional[ScriptStep]:
        """Move to the next step in the script."""
        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
            self.updated_at = datetime.now().isoformat()
            return self.steps[self.current_step_index]
        return None
    
    def update_step_status(
        self, 
        step_id: str, 
        status: str, 
        result: Optional[str] = None, 
        error: Optional[str] = None
    ) -> bool:
        """Update the status of a step.
        
        Args:
            step_id: ID of the step to update
            status: New status (pending, in_progress, completed, failed)
            result: Result of the step execution
            error: Error message if the step failed
            
        Returns:
            bool: True if the step was found and updated, False otherwise
        """
        for step in self.steps:
            if step.step_id == step_id:
                step.status = status
                
                if status == "in_progress" and not step.start_time:
                    step.start_time = datetime.now().isoformat()
                
                if status in ["completed", "failed"]:
                    step.end_time = datetime.now().isoformat()
                    
                if result is not None:
                    step.result = result
                    
                if error is not None:
                    step.error = error
                
                self.updated_at = datetime.now().isoformat()
                return True
        
        return False
    
    def get_progress(self) -> Dict[str, Any]:
        """Get the overall progress of the script execution."""
        total_steps = len(self.steps)
        completed_steps = sum(1 for step in self.steps if step.status == "completed")
        failed_steps = sum(1 for step in self.steps if step.status == "failed")
        in_progress_steps = sum(1 for step in self.steps if step.status == "in_progress")
        
        progress_percentage = 0
        if total_steps > 0:
            progress_percentage = (completed_steps / total_steps) * 100
            
        return {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "in_progress_steps": in_progress_steps,
            "progress_percentage": progress_percentage,
            "current_step_index": self.current_step_index,
            "current_step": self.get_current_step().to_dict() if self.get_current_step() else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert script to dictionary representation."""
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_step_index": self.current_step_index,
            "steps": [step.to_dict() for step in self.steps]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Script':
        """Create a script from dictionary representation."""
        script = cls(
            task_id=data["task_id"],
            task_description=data["task_description"]
        )
        script.created_at = data.get("created_at", script.created_at)
        script.updated_at = data.get("updated_at", script.updated_at)
        script.current_step_index = data.get("current_step_index", 0)
        
        for step_data in data.get("steps", []):
            script.steps.append(ScriptStep.from_dict(step_data))
            
        return script


class ScriptStorage:
    """Storage for scripts and their execution state."""
    
    def __init__(self, storage_dir: str = "./scripts"):
        """Initialize script storage.
        
        Args:
            storage_dir: Directory to store script files
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
    def save_script(self, script: Script) -> None:
        """Save a script to storage."""
        file_path = os.path.join(self.storage_dir, f"{script.task_id}.json")
        with open(file_path, "w") as f:
            json.dump(script.to_dict(), f, indent=2)
            
    def load_script(self, task_id: str) -> Optional[Script]:
        """Load a script from storage."""
        file_path = os.path.join(self.storage_dir, f"{task_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
                return Script.from_dict(data)
        return None
        
    def list_scripts(self) -> List[Dict[str, Any]]:
        """List all available scripts."""
        scripts = []
        for file_name in os.listdir(self.storage_dir):
            if file_name.endswith(".json"):
                task_id = file_name[:-5]
                script = self.load_script(task_id)
                if script:
                    scripts.append(script.to_dict())
        return scripts
    
    def delete_script(self, task_id: str) -> bool:
        """Delete a script from storage."""
        file_path = os.path.join(self.storage_dir, f"{task_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False


class ScriptGenerator:
    """Generates execution scripts from task descriptions."""
    
    def __init__(self, model=None):
        """Initialize script generator.
        
        Args:
            model: LLM model to use for script generation
        """
        self.model = model
        
    def generate_script(self, task_description: str) -> Script:
        """Generate a script from a task description.
        
        Args:
            task_description: Description of the task to generate a script for
            
        Returns:
            Script: Generated script with steps
        """
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        
        # Create a new script
        script = Script(task_id=task_id, task_description=task_description)
        
        # If we have a model, use it to generate steps
        if self.model:
            # This would be implemented with actual LLM calls
            # For now, we'll create a simple placeholder script
            steps = self._generate_steps_with_model(task_description)
            for step_data in steps:
                script.add_step(ScriptStep(**step_data))
        else:
            # Create a simple default script with placeholder steps
            script.add_step(ScriptStep(
                step_id="step_1",
                description="Analyze the task requirements",
                tools=["search", "browser"],
                expected_output="Clear understanding of task requirements"
            ))
            script.add_step(ScriptStep(
                step_id="step_2",
                description="Gather necessary information",
                tools=["search", "browser"],
                expected_output="Collected information needed for the task"
            ))
            script.add_step(ScriptStep(
                step_id="step_3",
                description="Execute the required actions",
                tools=["terminal", "file_write"],
                expected_output="Actions completed successfully"
            ))
            script.add_step(ScriptStep(
                step_id="step_4",
                description="Verify results and finalize",
                tools=["browser", "terminal"],
                expected_output="Verified results and final output"
            ))
        
        return script
    
    def _generate_steps_with_model(self, task_description: str) -> List[Dict[str, Any]]:
        """Generate script steps using an LLM model.
        
        This is a placeholder implementation. In a real implementation,
        this would make calls to an LLM to generate the steps.
        """
        # Placeholder implementation
        return [
            {
                "step_id": "step_1",
                "description": "Analyze the task requirements",
                "tools": ["search", "browser"],
                "expected_output": "Clear understanding of task requirements"
            },
            {
                "step_id": "step_2",
                "description": "Gather necessary information",
                "tools": ["search", "browser"],
                "expected_output": "Collected information needed for the task"
            },
            {
                "step_id": "step_3",
                "description": "Execute the required actions",
                "tools": ["terminal", "file_write"],
                "expected_output": "Actions completed successfully"
            },
            {
                "step_id": "step_4",
                "description": "Verify results and finalize",
                "tools": ["browser", "terminal"],
                "expected_output": "Verified results and final output"
            }
        ]
    
    def update_script(self, script: Script, feedback: str) -> Script:
        """Update a script based on feedback.
        
        Args:
            script: Existing script to update
            feedback: Feedback to incorporate into the script
            
        Returns:
            Script: Updated script
        """
        # This would be implemented with actual LLM calls
        # For now, we'll just add a new step based on the feedback
        script.add_step(ScriptStep(
            step_id=f"step_{len(script.steps) + 1}",
            description=f"Address feedback: {feedback}",
            tools=["terminal", "browser"],
            expected_output="Feedback addressed successfully"
        ))
        
        return script


class ProgressTracker:
    """Tracks and reports progress of script execution."""
    
    def __init__(self, script_storage: ScriptStorage):
        """Initialize progress tracker.
        
        Args:
            script_storage: Storage for scripts
        """
        self.script_storage = script_storage
        self.update_callbacks = []
        
    def get_overall_progress(self, task_id: str) -> Dict[str, Any]:
        """Get the overall progress of a script.
        
        Args:
            task_id: ID of the task to get progress for
            
        Returns:
            Dict: Progress information
        """
        script = self.script_storage.load_script(task_id)
        if script:
            return script.get_progress()
        return {
            "error": "Script not found",
            "task_id": task_id
        }
    
    def get_step_progress(self, task_id: str, step_id: str) -> Dict[str, Any]:
        """Get the progress of a specific step.
        
        Args:
            task_id: ID of the task
            step_id: ID of the step
            
        Returns:
            Dict: Step information
        """
        script = self.script_storage.load_script(task_id)
        if not script:
            return {"error": "Script not found", "task_id": task_id}
        
        for step in script.steps:
            if step.step_id == step_id:
                return step.to_dict()
                
        return {"error": "Step not found", "task_id": task_id, "step_id": step_id}
    
    def update_progress(
        self, 
        task_id: str, 
        step_id: str, 
        status: str, 
        result: Optional[str] = None, 
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update the progress of a step.
        
        Args:
            task_id: ID of the task
            step_id: ID of the step
            status: New status
            result: Result of the step execution
            error: Error message if the step failed
            
        Returns:
            Dict: Updated progress information
        """
        script = self.script_storage.load_script(task_id)
        if not script:
            return {"error": "Script not found", "task_id": task_id}
        
        updated = script.update_step_status(step_id, status, result, error)
        if not updated:
            return {"error": "Step not found", "task_id": task_id, "step_id": step_id}
        
        # If step is completed, automatically move to next step
        if status == "completed":
            current_step = script.get_current_step()
            if current_step and current_step.step_id == step_id:
                script.move_to_next_step()
        
        # Save the updated script
        self.script_storage.save_script(script)
        
        # Notify callbacks
        progress = script.get_progress()
        for callback in self.update_callbacks:
            try:
                callback(task_id, progress)
            except Exception as e:
                logger.error(f"Error in progress update callback: {str(e)}")
        
        return progress
    
    def subscribe_to_updates(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Subscribe to progress updates.
        
        Args:
            callback: Function to call when progress is updated
        """
        self.update_callbacks.append(callback)
    
    def generate_progress_report(self, task_id: str) -> Dict[str, Any]:
        """Generate a detailed progress report.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dict: Detailed progress report
        """
        script = self.script_storage.load_script(task_id)
        if not script:
            return {"error": "Script not found", "task_id": task_id}
        
        progress = script.get_progress()
        
        # Add more detailed information
        progress["task_description"] = script.task_description
        progress["created_at"] = script.created_at
        progress["updated_at"] = script.updated_at
        progress["steps"] = [step.to_dict() for step in script.steps]
        
        # Calculate time metrics
        completed_steps = [step for step in script.steps if step.status == "completed" and step.start_time and step.end_time]
        if completed_steps:
            total_duration = sum(
                (datetime.fromisoformat(step.end_time) - datetime.fromisoformat(step.start_time)).total_seconds()
                for step in completed_steps
            )
            avg_step_duration = total_duration / len(completed_steps)
            progress["avg_step_duration_seconds"] = avg_step_duration
            
            # Estimate remaining time
            remaining_steps = len(script.steps) - progress["completed_steps"] - progress["failed_steps"]
            if remaining_steps > 0:
                progress["estimated_remaining_seconds"] = avg_step_duration * remaining_steps
        
        return progress


class ScriptManager:
    """Manages scripts and their execution."""
    
    def __init__(
        self, 
        script_storage: Optional[ScriptStorage] = None,
        script_generator: Optional[ScriptGenerator] = None,
        progress_tracker: Optional[ProgressTracker] = None
    ):
        """Initialize script manager.
        
        Args:
            script_storage: Storage for scripts
            script_generator: Generator for scripts
            progress_tracker: Tracker for script progress
        """
        self.script_storage = script_storage or ScriptStorage()
        self.script_generator = script_generator or ScriptGenerator()
        self.progress_tracker = progress_tracker or ProgressTracker(self.script_storage)
        
    def create_script(self, task_description: str) -> str:
        """Create a new script.
        
        Args:
            task_description: Description of the task
            
        Returns:
            str: ID of the created script
        """
        script = self.script_generator.generate_script(task_description)
        self.script_storage.save_script(script)
        return script.task_id
    
    def get_script(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a script by ID.
        
        Args:
            task_id: ID of the script
            
        Returns:
            Dict: Script data or None if not found
        """
        script = self.script_storage.load_script(task_id)
        if script:
            return script.to_dict()
        return None
    
    def update_script(self, task_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a script.
        
        Args:
            task_id: ID of the script
            updates: Updates to apply
            
        Returns:
            Dict: Updated script data or None if not found
        """
        script = self.script_storage.load_script(task_id)
        if not script:
            return None
        
        # Apply updates
        if "task_description" in updates:
            script.task_description = updates["task_description"]
            
        if "current_step_index" in updates:
            script.current_step_index = updates["current_step_index"]
            
        if "steps" in updates:
            # Replace steps if provided
            script.steps = []
            for step_data in updates["steps"]:
                script.steps.append(ScriptStep.from_dict(step_data))
        
        script.updated_at = datetime.now().isoformat()
        self.script_storage.save_script(script)
        return script.to_dict()
    
    def get_current_step(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the current step of a script.
        
        Args:
            task_id: ID of the script
            
        Returns:
            Dict: Current step data or None if not found
        """
        script = self.script_storage.load_script(task_id)
        if script:
            current_step = script.get_current_step()
            if current_step:
                return current_step.to_dict()
        return None
    
    def move_to_next_step(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Move to the next step of a script.
        
        Args:
            task_id: ID of the script
            
        Returns:
            Dict: Next step data or None if at the end or script not found
        """
        script = self.script_storage.load_script(task_id)
        if not script:
            return None
        
        next_step = script.move_to_next_step()
        if next_step:
            self.script_storage.save_script(script)
            return next_step.to_dict()
        return None
    
    def update_step_status(
        self, 
        task_id: str, 
        step_id: str, 
        status: str, 
        result: Optional[str] = None, 
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update the status of a step.
        
        Args:
            task_id: ID of the script
            step_id: ID of the step
            status: New status
            result: Result of the step execution
            error: Error message if the step failed
            
        Returns:
            Dict: Updated progress information
        """
        return self.progress_tracker.update_progress(task_id, step_id, status, result, error)
    
    def regenerate_script(self, task_id: str, feedback: Optional[str] = None) -> Optional[str]:
        """Regenerate a script based on feedback.
        
        Args:
            task_id: ID of the script
            feedback: Feedback to incorporate
            
        Returns:
            str: ID of the regenerated script or None if original not found
        """
        original_script = self.script_storage.load_script(task_id)
        if not original_script:
            return None
        
        # Generate a new script based on the original task
        new_script = self.script_generator.generate_script(original_script.task_description)
        
        # If feedback is provided, update the script
        if feedback:
            new_script = self.script_generator.update_script(new_script, feedback)
        
        # Save the new script
        self.script_storage.save_script(new_script)
        return new_script.task_id
    
    def list_scripts(self) -> List[Dict[str, Any]]:
        """List all available scripts.
        
        Returns:
            List[Dict]: List of script summaries
        """
        return self.script_storage.list_scripts()
    
    def delete_script(self, task_id: str) -> bool:
        """Delete a script.
        
        Args:
            task_id: ID of the script
            
        Returns:
            bool: True if deleted, False if not found
        """
        return self.script_storage.delete_script(task_id)
