#!/usr/bin/env python3

import os
import shlex
import subprocess
import tempfile
from typing import List, Optional
from pathlib import Path

from aider.coders.base_coder import Coder
from aider.io import InputOutput


class CodexCoder(Coder):
    """
    Codex agent for Aider benchmarking.
    Uses Codex CLI to solve coding problems through terminal interaction.
    """
    
    edit_format = "codex"

    def __init__(self, main_model, io: InputOutput, model_name: str = "gpt-4o", **kwargs):
        super().__init__(main_model, io, **kwargs)
        self.model_name = model_name
        self.conversational = True

        # Set up environment for Codex CLI
        self.env = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        }

        # Check if codex command is available
        try:
            result = subprocess.run(["codex", "--version"], capture_output=True, check=True)
            self.io.tool_output(f"Codex CLI version: {result.stdout.decode().strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "Codex CLI not found. Please ensure it's installed and available in PATH.\n"
                "Install with: npm install -g @openai/codex"
            ) from e

    def _create_workspace_prompt(self, user_message: str, files_content: dict, test_failures: str = None) -> str:
        """Create a comprehensive prompt using Aider's format."""
        
        # Build file list for instructions
        file_list = ", ".join(files_content.keys()) if files_content else "the supplied files"
        
        # Start with the user message (problem statement or instructions)
        prompt = user_message.strip() + "\n"
        
        if test_failures:
            # If we have test failures, use the test failure template
            test_failures_addendum = f"""
####
See the testing errors above.
The tests are correct, don't try and change them.
Fix the code in {file_list} to resolve the errors.
"""
            prompt += test_failures_addendum
        else:
            # Use the standard instructions addendum
            instructions_addendum = f"""
####
Use the above instructions to modify the supplied files: {file_list}
Don't change the names of existing functions or classes, as they may be referenced from other code like unit tests, etc.
Only use standard libraries, don't suggest installing any packages.
"""
            prompt += instructions_addendum
            
        return prompt

    def _setup_workspace(self, files: dict) -> str:
        """Set up temporary workspace with provided files."""
        work_dir = tempfile.mkdtemp()
        
        for filename, content in files.items():
            file_path = Path(work_dir) / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            
        return work_dir

    def _run_codex_command(self, prompt: str, work_dir: str) -> str:
        """Run codex command with the prompt in the working directory."""
        escaped_prompt = shlex.quote(prompt)
        cmd = [
            "codex",
            "--writable-root", "/",
            "-q", 
            "--approval-mode", "full-auto",
            "--model", self.model_name,
            escaped_prompt
        ]

        # Run the command in the working directory
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=work_dir,
            env={**os.environ, **self.env}
        )

        if result.returncode != 0:
            self.io.tool_error(f"Codex command failed: {result.stderr}")
            raise RuntimeError(f"Codex command failed: {result.stderr}")

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
                except Exception as e:
                    self.io.tool_error(f"Error reading {filename}: {e}")
                    continue
                    
        return modified_files

    def send_new_user_message(self, inp: str, files_content: Optional[dict] = None):
        """
        Send a new user message through Codex CLI.
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
                    self.io.tool_error(f"Error reading {fname}: {e}")
                    files_content[os.path.basename(fname)] = ""

        work_dir = None

        try:
            # Setup workspace with current files
            work_dir = self._setup_workspace(files_content)
            self.io.tool_output(f"Set up workspace: {work_dir}")

            # Create comprehensive prompt using Aider's format
            full_prompt = self._create_workspace_prompt(inp, files_content)

            # Run Codex CLI
            output = self._run_codex_command(full_prompt, work_dir)
            self.io.tool_output("Codex execution completed")

            # Extract modified files
            modified_files = self._extract_modified_files(work_dir, files_content)
            
            if modified_files:
                self.io.tool_output(f"Modified {len(modified_files)} files")
                
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
                            self.io.tool_output(f"Updated {full_path}")
                        except Exception as e:
                            self.io.tool_error(f"Error writing {full_path}: {e}")
                    else:
                        self.io.tool_error(f"Could not find full path for {filename}")
            else:
                self.io.tool_output("No files were modified")
                
            # Track outcomes for benchmarking
            self.edit_outcome = len(modified_files) > 0
            self.lint_outcome = True  # Assume success for now
            self.test_outcome = None  # Will be determined by test runner

        except Exception as e:
            self.io.tool_error(f"Error in Codex processing: {e}")
            self.edit_outcome = False
            self.lint_outcome = False
            self.test_outcome = False
            raise

        finally:
            # Clean up temporary workspace
            if work_dir:
                try:
                    import shutil
                    shutil.rmtree(work_dir)
                except OSError:
                    pass

    def send_test_failure_message(self, test_error_output: str, files_content: Optional[dict] = None):
        """
        Handle test failures using Aider's test failure prompt format.
        """
        if not files_content:
            # Get current files content from the repository
            files_content = {}
            for fname in self.abs_fnames:
                try:
                    with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
                        files_content[os.path.basename(fname)] = f.read()
                except Exception as e:
                    self.io.tool_error(f"Error reading {fname}: {e}")
                    files_content[os.path.basename(fname)] = ""

        work_dir = None

        try:
            # Setup workspace with current files
            work_dir = self._setup_workspace(files_content)
            
            # Create prompt with test failure format
            full_prompt = self._create_workspace_prompt(test_error_output, files_content, test_failures=test_error_output)

            # Run Codex CLI
            output = self._run_codex_command(full_prompt, work_dir)
            
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
                        with open(full_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        self.io.tool_output(f"Fixed {full_path}")

            self.edit_outcome = len(modified_files) > 0

        except Exception as e:
            self.io.tool_error(f"Error fixing test failures: {e}")
            self.edit_outcome = False
            raise
            
        finally:
            # Clean up
            if work_dir:
                try:
                    import shutil
                    shutil.rmtree(work_dir)
                except OSError:
                    pass

    def run_with_retries(self, user_message: str, max_retries: int = 3):
        """Run the coder with retries for robustness."""
        for attempt in range(max_retries):
            try:
                self.send_new_user_message(user_message)
                if self.edit_outcome:
                    break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.io.tool_output(f"Attempt {attempt + 1} failed, retrying: {e}")
                continue
