#!/usr/bin/env python3

import os
import shlex
import subprocess
import tempfile
from typing import List, Optional
from pathlib import Path

from aider.coders.base_coder import Coder
from aider.io import InputOutput


class ClaudeCodeCoder(Coder):
    """
    Claude Code agent for Aider benchmarking.
    Uses Claude Code CLI with raw prompt format (matching TBench).
    """
    
    edit_format = "claude-code"
    
    ALLOWED_TOOLS = [
        "Bash",
        "Edit", 
        "Write",
        "Read",
        "Glob",
        "Grep",
        "LS",
        "WebFetch",
        "NotebookEdit",
        "NotebookRead",
        "TodoRead",
        "TodoWrite",
        "Agent",
    ]

    def __init__(self, main_model, io: InputOutput, model_name: str = "claude-3-5-sonnet-20241022", **kwargs):
        super().__init__(main_model, io, **kwargs)
        self.model_name = model_name
        self.conversational = True

        # Set up environment for Claude Code CLI - only add non-None values
        self.env = {}
        
        # Only add API key if it exists
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            self.env["ANTHROPIC_API_KEY"] = api_key
        
        # These are always set
        self.env["FORCE_AUTO_BACKGROUND_TASKS"] = "1"
        self.env["ENABLE_BACKGROUND_TASKS"] = "1"

        # Check if claude command is available
        try:
            result = subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "Claude Code CLI not found. Please ensure it's installed and available in PATH.\n"
                "Install with: npm install -g @anthropic-ai/claude-code"
            ) from e

    def _create_prompt_file(self, user_message: str, files_content: dict, test_failures: str = None) -> str:
        """Create a temporary file with raw instruction (matching TBench format)."""
        
        # Use raw instruction format - no Aider formatting
        if test_failures:
            # For test failures, just pass the error message
            full_prompt = test_failures.strip()
        else:
            # For initial instructions, just pass them raw
            full_prompt = user_message.strip()

        # Create temporary file and return the path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(full_prompt)
            return f.name

    def _setup_workspace(self, files: dict) -> str:
        """Set up temporary workspace with provided files."""
        work_dir = tempfile.mkdtemp()
        
        for filename, content in files.items():
            file_path = Path(work_dir) / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            
        return work_dir

    def _run_claude_command(self, prompt_file: str, work_dir: str) -> str:
        """Run claude command matching TBench format exactly."""
        
        # Read the prompt content to pass as argument (like TBench does)
        with open(prompt_file, 'r') as f:
            instruction = f.read()
        
        escaped_instruction = shlex.quote(instruction)
        
        cmd = [
            "claude", 
            "--verbose", 
            "--output-format", "stream-json",
            "-p", escaped_instruction,
            "--allowedTools", " ".join(self.ALLOWED_TOOLS)
        ]
        
        # Simple environment merge - no None values
        env_vars = {**os.environ, **self.env}
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=work_dir,
            env=env_vars,
            timeout=300
        )

        if result.returncode != 0:
            error_msg = f"Claude command failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f": {result.stderr}"
            raise RuntimeError(error_msg)

        return result.stdout

    def _extract_modified_files(self, work_dir: str, original_files: dict) -> dict:
        """Extract modified files from the working directory."""
        modified_files = {}
        
        for filename in original_files.keys():
            file_path = Path(work_dir) / filename
            if file_path.exists():
                try:
                    new_content = file_path.read_text()
                    if new_content != original_files[filename]:
                        modified_files[filename] = new_content
                except Exception:
                    continue
                    
        return modified_files

    def format_messages(self):
        """Override to prevent chat formatting - Claude Code doesn't use chat messages."""
        return []

    def run_one(self, with_message, preproc=False):
        """Override to use our direct file-based approach instead of chat."""
        self.send_new_user_message(with_message)
        return []

    def send_message(self, message):
        """Override to use our direct approach."""
        self.send_new_user_message(message)
        return []

    def send_new_user_message(self, inp: str, files_content: Optional[dict] = None):
        """
        Send a new user message through Claude Code CLI.
        This is the main method called by the Aider benchmark.
        """
        if not files_content:
            # Get current files content from the repository
            files_content = {}
            for fname in self.abs_fnames:
                try:
                    with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
                        files_content[os.path.basename(fname)] = f.read()
                except Exception as e:
                    files_content[os.path.basename(fname)] = ""

        prompt_file = None
        work_dir = None

        try:
            # Setup workspace with current files
            work_dir = self._setup_workspace(files_content)

            # Create prompt file using raw format (matching TBench)
            prompt_file = self._create_prompt_file(inp, files_content)

            # Run Claude Code CLI
            output = self._run_claude_command(prompt_file, work_dir)

            # Extract modified files
            modified_files = self._extract_modified_files(work_dir, files_content)
            
            if modified_files:
                # Apply changes to actual files
                for filename, new_content in modified_files.items():
                    # Find the full path for this file
                    full_path = None
                    for fname in self.abs_fnames:
                        if os.path.basename(fname) == filename:
                            full_path = fname
                            break
                    
                    if full_path:
                        try:
                            with open(full_path, 'w', encoding='utf-8') as f:
                                f.write(new_content)
                        except Exception:
                            pass
                
            # Track outcomes for benchmarking
            self.edit_outcome = len(modified_files) > 0
            self.lint_outcome = True
            self.test_outcome = None

        except Exception:
            self.edit_outcome = False
            self.lint_outcome = False
            self.test_outcome = False
            raise

        finally:
            # Clean up temporary files
            if prompt_file:
                try:
                    os.unlink(prompt_file)
                except OSError:
                    pass

            if work_dir:
                try:
                    import shutil
                    shutil.rmtree(work_dir)
                except OSError:
                    pass

    def send_test_failure_message(self, test_error_output: str, files_content: Optional[dict] = None):
        """
        Handle test failures using raw format.
        """
        if not files_content:
            # Get current files content from the repository
            files_content = {}
            for fname in self.abs_fnames:
                try:
                    with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
                        files_content[os.path.basename(fname)] = f.read()
                except Exception:
                    files_content[os.path.basename(fname)] = ""

        prompt_file = None
        work_dir = None

        try:
            # Setup workspace with current files
            work_dir = self._setup_workspace(files_content)
            
            # Create prompt file with test failure format (raw)
            prompt_file = self._create_prompt_file("", files_content, test_failures=test_error_output)

            # Run Claude Code CLI
            output = self._run_claude_command(prompt_file, work_dir)
            
            # Extract and apply modified files
            modified_files = self._extract_modified_files(work_dir, files_content)
            
            if modified_files:
                for filename, new_content in modified_files.items():
                    full_path = None
                    for fname in self.abs_fnames:
                        if os.path.basename(fname) == filename:
                            full_path = fname
                            break
                    
                    if full_path:
                        try:
                            with open(full_path, 'w', encoding='utf-8') as f:
                                f.write(new_content)
                        except Exception:
                            pass

            self.edit_outcome = len(modified_files) > 0

        except Exception:
            self.edit_outcome = False
            raise
            
        finally:
            # Clean up
            if prompt_file:
                try:
                    os.unlink(prompt_file)
                except OSError:
                    pass
            if work_dir:
                try:
                    import shutil
                    shutil.rmtree(work_dir)
                except OSError:
                    pass
