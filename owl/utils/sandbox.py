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
import subprocess
import tempfile
import shutil
import uuid
import time
from typing import Dict, List, Optional, Any, Union, Tuple

logger = logging.getLogger(__name__)

class SandboxEnvironment:
    """Represents a sandbox environment for safe execution of code and commands."""
    
    def __init__(
        self,
        sandbox_id: Optional[str] = None,
        base_dir: Optional[str] = None,
        max_memory_mb: int = 512,
        max_cpu_time: int = 30,
        max_disk_space_mb: int = 100,
    ):
        """Initialize a sandbox environment.
        
        Args:
            sandbox_id: Unique identifier for the sandbox
            base_dir: Base directory for sandbox files
            max_memory_mb: Maximum memory usage in MB
            max_cpu_time: Maximum CPU time in seconds
            max_disk_space_mb: Maximum disk space in MB
        """
        self.sandbox_id = sandbox_id or str(uuid.uuid4())
        self.base_dir = base_dir or os.path.join(tempfile.gettempdir(), "owl_sandbox")
        self.sandbox_dir = os.path.join(self.base_dir, self.sandbox_id)
        self.max_memory_mb = max_memory_mb
        self.max_cpu_time = max_cpu_time
        self.max_disk_space_mb = max_disk_space_mb
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.status = "created"  # created, active, inactive, terminated
        
        # Create sandbox directory
        os.makedirs(self.sandbox_dir, exist_ok=True)
        os.makedirs(os.path.join(self.sandbox_dir, "files"), exist_ok=True)
        os.makedirs(os.path.join(self.sandbox_dir, "logs"), exist_ok=True)
        
        # Initialize sandbox
        self._initialize_sandbox()
        
    def _initialize_sandbox(self):
        """Initialize the sandbox environment."""
        # Create a metadata file
        metadata = {
            "sandbox_id": self.sandbox_id,
            "created_at": self.created_at,
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_time": self.max_cpu_time,
            "max_disk_space_mb": self.max_disk_space_mb,
            "status": self.status
        }
        
        with open(os.path.join(self.sandbox_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        self.status = "active"
        self._update_metadata()
        
    def _update_metadata(self):
        """Update the sandbox metadata file."""
        metadata_path = os.path.join(self.sandbox_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}
            
        metadata.update({
            "last_accessed": self.last_accessed,
            "status": self.status
        })
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
    def execute_code(
        self, 
        code: str, 
        language: str = "python",
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute code in the sandbox.
        
        Args:
            code: Code to execute
            language: Programming language
            timeout: Execution timeout in seconds
            
        Returns:
            Dict: Execution result
        """
        self.last_accessed = time.time()
        
        if self.status != "active":
            return {
                "success": False,
                "error": f"Sandbox is not active (current status: {self.status})"
            }
            
        # Set timeout
        timeout = timeout or self.max_cpu_time
        
        # Create a file for the code
        file_ext = self._get_file_extension(language)
        code_file = os.path.join(self.sandbox_dir, "files", f"code_{int(time.time())}{file_ext}")
        
        with open(code_file, "w") as f:
            f.write(code)
            
        # Execute the code
        result = self._execute_file(code_file, language, timeout)
        
        # Update metadata
        self._update_metadata()
        
        return result
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for a language.
        
        Args:
            language: Programming language
            
        Returns:
            str: File extension
        """
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "node": ".js",
            "bash": ".sh",
            "shell": ".sh",
            "ruby": ".rb",
            "perl": ".pl",
            "php": ".php",
            "r": ".r",
            "go": ".go",
            "rust": ".rs",
            "java": ".java",
            "c": ".c",
            "cpp": ".cpp",
            "csharp": ".cs",
        }
        
        return extensions.get(language.lower(), ".txt")
    
    def _execute_file(
        self, 
        file_path: str, 
        language: str, 
        timeout: int
    ) -> Dict[str, Any]:
        """Execute a file in the sandbox.
        
        Args:
            file_path: Path to the file
            language: Programming language
            timeout: Execution timeout in seconds
            
        Returns:
            Dict: Execution result
        """
        # Get command to execute the file
        command = self._get_execution_command(file_path, language)
        
        if not command:
            return {
                "success": False,
                "error": f"Unsupported language: {language}"
            }
            
        # Create log files
        stdout_file = os.path.join(self.sandbox_dir, "logs", f"stdout_{int(time.time())}.txt")
        stderr_file = os.path.join(self.sandbox_dir, "logs", f"stderr_{int(time.time())}.txt")
        
        # Execute the command
        try:
            with open(stdout_file, "w") as stdout, open(stderr_file, "w") as stderr:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=stdout,
                    stderr=stderr,
                    cwd=os.path.join(self.sandbox_dir, "files"),
                    env=self._get_safe_env()
                )
                
                try:
                    exit_code = process.wait(timeout=timeout)
                    
                    # Read output
                    with open(stdout_file, "r") as f:
                        stdout_content = f.read()
                        
                    with open(stderr_file, "r") as f:
                        stderr_content = f.read()
                        
                    return {
                        "success": exit_code == 0,
                        "exit_code": exit_code,
                        "stdout": stdout_content,
                        "stderr": stderr_content,
                        "language": language,
                        "file": os.path.basename(file_path)
                    }
                    
                except subprocess.TimeoutExpired:
                    process.kill()
                    return {
                        "success": False,
                        "error": f"Execution timed out after {timeout} seconds",
                        "language": language,
                        "file": os.path.basename(file_path)
                    }
                    
        except Exception as e:
            logger.error(f"Error executing file: {str(e)}")
            return {
                "success": False,
                "error": f"Error executing file: {str(e)}",
                "language": language,
                "file": os.path.basename(file_path)
            }
    
    def _get_execution_command(self, file_path: str, language: str) -> Optional[str]:
        """Get command to execute a file.
        
        Args:
            file_path: Path to the file
            language: Programming language
            
        Returns:
            str: Execution command
        """
        commands = {
            "python": f"python3 {file_path}",
            "javascript": f"node {file_path}",
            "node": f"node {file_path}",
            "bash": f"bash {file_path}",
            "shell": f"bash {file_path}",
            "ruby": f"ruby {file_path}",
            "perl": f"perl {file_path}",
            "php": f"php {file_path}",
            "r": f"Rscript {file_path}",
            "go": f"go run {file_path}",
        }
        
        return commands.get(language.lower())
    
    def _get_safe_env(self) -> Dict[str, str]:
        """Get a safe environment for execution.
        
        Returns:
            Dict: Environment variables
        """
        # Start with a minimal environment
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": self.sandbox_dir,
            "SANDBOX_ID": self.sandbox_id,
            "SANDBOX_DIR": self.sandbox_dir,
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        
        return env
    
    def execute_command(
        self, 
        command: str,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute a shell command in the sandbox.
        
        Args:
            command: Command to execute
            timeout: Execution timeout in seconds
            
        Returns:
            Dict: Execution result
        """
        self.last_accessed = time.time()
        
        if self.status != "active":
            return {
                "success": False,
                "error": f"Sandbox is not active (current status: {self.status})"
            }
            
        # Set timeout
        timeout = timeout or self.max_cpu_time
        
        # Check if command is allowed
        if not self._is_command_allowed(command):
            return {
                "success": False,
                "error": f"Command not allowed: {command}"
            }
            
        # Create log files
        stdout_file = os.path.join(self.sandbox_dir, "logs", f"stdout_{int(time.time())}.txt")
        stderr_file = os.path.join(self.sandbox_dir, "logs", f"stderr_{int(time.time())}.txt")
        
        # Execute the command
        try:
            with open(stdout_file, "w") as stdout, open(stderr_file, "w") as stderr:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=stdout,
                    stderr=stderr,
                    cwd=os.path.join(self.sandbox_dir, "files"),
                    env=self._get_safe_env()
                )
                
                try:
                    exit_code = process.wait(timeout=timeout)
                    
                    # Read output
                    with open(stdout_file, "r") as f:
                        stdout_content = f.read()
                        
                    with open(stderr_file, "r") as f:
                        stderr_content = f.read()
                        
                    return {
                        "success": exit_code == 0,
                        "exit_code": exit_code,
                        "stdout": stdout_content,
                        "stderr": stderr_content,
                        "command": command
                    }
                    
                except subprocess.TimeoutExpired:
                    process.kill()
                    return {
                        "success": False,
                        "error": f"Execution timed out after {timeout} seconds",
                        "command": command
                    }
                    
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            return {
                "success": False,
                "error": f"Error executing command: {str(e)}",
                "command": command
            }
    
    def _is_command_allowed(self, command: str) -> bool:
        """Check if a command is allowed.
        
        Args:
            command: Command to check
            
        Returns:
            bool: True if allowed, False otherwise
        """
        # List of disallowed commands
        disallowed = [
            "rm -rf /",
            "sudo",
            "su",
            "chmod",
            "chown",
            "passwd",
            "mkfs",
            "dd",
            "mount",
            "umount",
            "apt",
            "apt-get",
            "yum",
            "dnf",
            "pacman",
            "systemctl",
            "service",
            "iptables",
            "firewall-cmd",
            "ufw",
            "ssh",
            "scp",
            "rsync",
            "nc",
            "ncat",
            "curl",
            "wget",
            "ftp",
            "telnet",
        ]
        
        # Check if command contains any disallowed commands
        for cmd in disallowed:
            if cmd in command:
                return False
                
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the sandbox.
        
        Returns:
            Dict: Sandbox status
        """
        self.last_accessed = time.time()
        
        # Check if sandbox directory exists
        if not os.path.exists(self.sandbox_dir):
            return {
                "sandbox_id": self.sandbox_id,
                "status": "terminated",
                "error": "Sandbox directory does not exist"
            }
            
        # Check if metadata file exists
        metadata_path = os.path.join(self.sandbox_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            return {
                "sandbox_id": self.sandbox_id,
                "status": "terminated",
                "error": "Metadata file does not exist"
            }
            
        # Read metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            
        # Get disk usage
        disk_usage = self._get_disk_usage()
        
        # Update metadata
        metadata.update({
            "last_accessed": self.last_accessed,
            "status": self.status,
            "disk_usage_mb": disk_usage
        })
        
        return metadata
    
    def _get_disk_usage(self) -> float:
        """Get disk usage of the sandbox.
        
        Returns:
            float: Disk usage in MB
        """
        total_size = 0
        
        for dirpath, dirnames, filenames in os.walk(self.sandbox_dir):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
                
        return total_size / (1024 * 1024)  # Convert to MB
    
    def cleanup(self) -> Dict[str, Any]:
        """Clean up the sandbox.
        
        Returns:
            Dict: Cleanup result
        """
        self.status = "terminated"
        self._update_metadata()
        
        try:
            shutil.rmtree(self.sandbox_dir)
            return {
                "success": True,
                "sandbox_id": self.sandbox_id,
                "status": "terminated"
            }
        except Exception as e:
            logger.error(f"Error cleaning up sandbox: {str(e)}")
            return {
                "success": False,
                "error": f"Error cleaning up sandbox: {str(e)}",
                "sandbox_id": self.sandbox_id,
                "status": self.status
            }
    
    def write_file(self, file_content: str, file_name: str) -> Dict[str, Any]:
        """Write content to a file in the sandbox.
        
        Args:
            file_content: Content to write
            file_name: Name of the file
            
        Returns:
            Dict: Write result
        """
        self.last_accessed = time.time()
        
        if self.status != "active":
            return {
                "success": False,
                "error": f"Sandbox is not active (current status: {self.status})"
            }
            
        # Ensure file name is safe
        file_name = os.path.basename(file_name)
        file_path = os.path.join(self.sandbox_dir, "files", file_name)
        
        try:
            with open(file_path, "w") as f:
                f.write(file_content)
                
            return {
                "success": True,
                "file": file_name,
                "path": file_path
            }
        except Exception as e:
            logger.error(f"Error writing file: {str(e)}")
            return {
                "success": False,
                "error": f"Error writing file: {str(e)}",
                "file": file_name
            }
    
    def read_file(self, file_name: str) -> Dict[str, Any]:
        """Read content from a file in the sandbox.
        
        Args:
            file_name: Name of the file
            
        Returns:
            Dict: Read result
        """
        self.last_accessed = time.time()
        
        # Ensure file name is safe
        file_name = os.path.basename(file_name)
        file_path = os.path.join(self.sandbox_dir, "files", file_name)
        
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File does not exist: {file_name}",
                "file": file_name
            }
            
        try:
            with open(file_path, "r") as f:
                content = f.read()
                
            return {
                "success": True,
                "file": file_name,
                "content": content
            }
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            return {
                "success": False,
                "error": f"Error reading file: {str(e)}",
                "file": file_name
            }
    
    def list_files(self) -> Dict[str, Any]:
        """List files in the sandbox.
        
        Returns:
            Dict: List result
        """
        self.last_accessed = time.time()
        
        files_dir = os.path.join(self.sandbox_dir, "files")
        
        try:
            files = []
            
            for file_name in os.listdir(files_dir):
                file_path = os.path.join(files_dir, file_name)
                
                if os.path.isfile(file_path):
                    files.append({
                        "name": file_name,
                        "size": os.path.getsize(file_path),
                        "modified": os.path.getmtime(file_path)
                    })
                    
            return {
                "success": True,
                "files": files
            }
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            return {
                "success": False,
                "error": f"Error listing files: {str(e)}"
            }


class SandboxManager:
    """Manages sandbox environments."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize sandbox manager.
        
        Args:
            base_dir: Base directory for sandbox files
        """
        self.base_dir = base_dir or os.path.join(tempfile.gettempdir(), "owl_sandbox")
        self.sandboxes = {}
        
        # Create base directory
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Load existing sandboxes
        self._load_existing_sandboxes()
        
    def _load_existing_sandboxes(self):
        """Load existing sandboxes from the base directory."""
        if not os.path.exists(self.base_dir):
            return
            
        for sandbox_id in os.listdir(self.base_dir):
            sandbox_dir = os.path.join(self.base_dir, sandbox_id)
            
            if os.path.isdir(sandbox_dir):
                metadata_path = os.path.join(sandbox_dir, "metadata.json")
                
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                            
                        if metadata.get("status") == "active":
                            self.sandboxes[sandbox_id] = SandboxEnvironment(
                                sandbox_id=sandbox_id,
                                base_dir=self.base_dir
                            )
                    except Exception as e:
                        logger.error(f"Error loading sandbox {sandbox_id}: {str(e)}")
    
    def create_sandbox(
        self,
        max_memory_mb: int = 512,
        max_cpu_time: int = 30,
        max_disk_space_mb: int = 100
    ) -> str:
        """Create a new sandbox environment.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            max_cpu_time: Maximum CPU time in seconds
            max_disk_space_mb: Maximum disk space in MB
            
        Returns:
            str: Sandbox ID
        """
        # Clean up old sandboxes
        self._cleanup_old_sandboxes()
        
        # Create a new sandbox
        sandbox = SandboxEnvironment(
            base_dir=self.base_dir,
            max_memory_mb=max_memory_mb,
            max_cpu_time=max_cpu_time,
            max_disk_space_mb=max_disk_space_mb
        )
        
        self.sandboxes[sandbox.sandbox_id] = sandbox
        return sandbox.sandbox_id
    
    def _cleanup_old_sandboxes(self):
        """Clean up old sandboxes."""
        current_time = time.time()
        
        for sandbox_id, sandbox in list(self.sandboxes.items()):
            # Check if sandbox is older than 1 hour
            if current_time - sandbox.created_at > 3600:
                sandbox.cleanup()
                del self.sandboxes[sandbox_id]
    
    def get_sandbox(self, sandbox_id: str) -> Optional[SandboxEnvironment]:
        """Get a sandbox by ID.
        
        Args:
            sandbox_id: Sandbox ID
            
        Returns:
            SandboxEnvironment: Sandbox environment
        """
        if sandbox_id in self.sandboxes:
            return self.sandboxes[sandbox_id]
            
        # Try to load the sandbox
        sandbox_dir = os.path.join(self.base_dir, sandbox_id)
        
        if os.path.exists(sandbox_dir):
            try:
                sandbox = SandboxEnvironment(
                    sandbox_id=sandbox_id,
                    base_dir=self.base_dir
                )
                
                self.sandboxes[sandbox_id] = sandbox
                return sandbox
            except Exception as e:
                logger.error(f"Error loading sandbox {sandbox_id}: {str(e)}")
                
        return None
    
    def execute_code(
        self, 
        sandbox_id: str, 
        code: str, 
        language: str = "python",
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute code in a sandbox.
        
        Args:
            sandbox_id: Sandbox ID
            code: Code to execute
            language: Programming language
            timeout: Execution timeout in seconds
            
        Returns:
            Dict: Execution result
        """
        sandbox = self.get_sandbox(sandbox_id)
        
        if not sandbox:
            return {
                "success": False,
                "error": f"Sandbox not found: {sandbox_id}"
            }
            
        return sandbox.execute_code(code, language, timeout)
    
    def execute_command(
        self, 
        sandbox_id: str, 
        command: str,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute a shell command in a sandbox.
        
        Args:
            sandbox_id: Sandbox ID
            command: Command to execute
            timeout: Execution timeout in seconds
            
        Returns:
            Dict: Execution result
        """
        sandbox = self.get_sandbox(sandbox_id)
        
        if not sandbox:
            return {
                "success": False,
                "error": f"Sandbox not found: {sandbox_id}"
            }
            
        return sandbox.execute_command(command, timeout)
    
    def get_sandbox_status(self, sandbox_id: str) -> Dict[str, Any]:
        """Get the status of a sandbox.
        
        Args:
            sandbox_id: Sandbox ID
            
        Returns:
            Dict: Sandbox status
        """
        sandbox = self.get_sandbox(sandbox_id)
        
        if not sandbox:
            return {
                "sandbox_id": sandbox_id,
                "status": "not_found",
                "error": f"Sandbox not found: {sandbox_id}"
            }
            
        return sandbox.get_status()
    
    def cleanup_sandbox(self, sandbox_id: str) -> Dict[str, Any]:
        """Clean up a sandbox.
        
        Args:
            sandbox_id: Sandbox ID
            
        Returns:
            Dict: Cleanup result
        """
        sandbox = self.get_sandbox(sandbox_id)
        
        if not sandbox:
            return {
                "success": False,
                "error": f"Sandbox not found: {sandbox_id}"
            }
            
        result = sandbox.cleanup()
        
        if result["success"]:
            del self.sandboxes[sandbox_id]
            
        return result
    
    def write_file(
        self, 
        sandbox_id: str, 
        file_content: str, 
        file_name: str
    ) -> Dict[str, Any]:
        """Write content to a file in a sandbox.
        
        Args:
            sandbox_id: Sandbox ID
            file_content: Content to write
            file_name: Name of the file
            
        Returns:
            Dict: Write result
        """
        sandbox = self.get_sandbox(sandbox_id)
        
        if not sandbox:
            return {
                "success": False,
                "error": f"Sandbox not found: {sandbox_id}"
            }
            
        return sandbox.write_file(file_content, file_name)
    
    def read_file(self, sandbox_id: str, file_name: str) -> Dict[str, Any]:
        """Read content from a file in a sandbox.
        
        Args:
            sandbox_id: Sandbox ID
            file_name: Name of the file
            
        Returns:
            Dict: Read result
        """
        sandbox = self.get_sandbox(sandbox_id)
        
        if not sandbox:
            return {
                "success": False,
                "error": f"Sandbox not found: {sandbox_id}"
            }
            
        return sandbox.read_file(file_name)
    
    def list_files(self, sandbox_id: str) -> Dict[str, Any]:
        """List files in a sandbox.
        
        Args:
            sandbox_id: Sandbox ID
            
        Returns:
            Dict: List result
        """
        sandbox = self.get_sandbox(sandbox_id)
        
        if not sandbox:
            return {
                "success": False,
                "error": f"Sandbox not found: {sandbox_id}"
            }
            
        return sandbox.list_files()
    
    def list_sandboxes(self) -> List[Dict[str, Any]]:
        """List all sandboxes.
        
        Returns:
            List[Dict]: List of sandbox statuses
        """
        sandboxes = []
        
        for sandbox_id, sandbox in self.sandboxes.items():
            sandboxes.append(sandbox.get_status())
            
        return sandboxes


class SandboxToolkit:
    """Toolkit for interacting with sandboxes."""
    
    def __init__(self, sandbox_manager: Optional[SandboxManager] = None):
        """Initialize sandbox toolkit.
        
        Args:
            sandbox_manager: Sandbox manager
        """
        self.sandbox_manager = sandbox_manager or SandboxManager()
        
    def get_tools(self) -> List[Any]:
        """Get all tools in the toolkit.
        
        Returns:
            List: List of tools
        """
        return [
            self.execute_code,
            self.execute_command,
            self.get_sandbox_status,
            self.cleanup_sandbox,
            self.write_file,
            self.read_file,
            self.list_files,
        ]
        
    def execute_code(
        self, 
        code: str, 
        language: str = "python",
        sandbox_id: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute code in a sandbox.
        
        Args:
            code: Code to execute
            language: Programming language
            sandbox_id: Sandbox ID (if None, a new sandbox will be created)
            timeout: Execution timeout in seconds
            
        Returns:
            Dict: Execution result
        """
        # Create a new sandbox if needed
        if not sandbox_id:
            sandbox_id = self.sandbox_manager.create_sandbox()
            
        # Execute the code
        result = self.sandbox_manager.execute_code(sandbox_id, code, language, timeout)
        result["sandbox_id"] = sandbox_id
        
        return result
    
    def execute_command(
        self, 
        command: str,
        sandbox_id: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute a shell command in a sandbox.
        
        Args:
            command: Command to execute
            sandbox_id: Sandbox ID (if None, a new sandbox will be created)
            timeout: Execution timeout in seconds
            
        Returns:
            Dict: Execution result
        """
        # Create a new sandbox if needed
        if not sandbox_id:
            sandbox_id = self.sandbox_manager.create_sandbox()
            
        # Execute the command
        result = self.sandbox_manager.execute_command(sandbox_id, command, timeout)
        result["sandbox_id"] = sandbox_id
        
        return result
    
    def get_sandbox_status(self, sandbox_id: str) -> Dict[str, Any]:
        """Get the status of a sandbox.
        
        Args:
            sandbox_id: Sandbox ID
            
        Returns:
            Dict: Sandbox status
        """
        return self.sandbox_manager.get_sandbox_status(sandbox_id)
    
    def cleanup_sandbox(self, sandbox_id: str) -> Dict[str, Any]:
        """Clean up a sandbox.
        
        Args:
            sandbox_id: Sandbox ID
            
        Returns:
            Dict: Cleanup result
        """
        return self.sandbox_manager.cleanup_sandbox(sandbox_id)
    
    def write_file(
        self, 
        file_content: str, 
        file_name: str,
        sandbox_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Write content to a file in a sandbox.
        
        Args:
            file_content: Content to write
            file_name: Name of the file
            sandbox_id: Sandbox ID (if None, a new sandbox will be created)
            
        Returns:
            Dict: Write result
        """
        # Create a new sandbox if needed
        if not sandbox_id:
            sandbox_id = self.sandbox_manager.create_sandbox()
            
        # Write the file
        result = self.sandbox_manager.write_file(sandbox_id, file_content, file_name)
        result["sandbox_id"] = sandbox_id
        
        return result
    
    def read_file(self, file_name: str, sandbox_id: str) -> Dict[str, Any]:
        """Read content from a file in a sandbox.
        
        Args:
            file_name: Name of the file
            sandbox_id: Sandbox ID
            
        Returns:
            Dict: Read result
        """
        return self.sandbox_manager.read_file(sandbox_id, file_name)
    
    def list_files(self, sandbox_id: str) -> Dict[str, Any]:
        """List files in a sandbox.
        
        Args:
            sandbox_id: Sandbox ID
            
        Returns:
            Dict: List result
        """
        return self.sandbox_manager.list_files(sandbox_id)
