# aider/coders/claude_code_coder.py
from __future__ import annotations

import json
import os
import shlex
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from aider.coders.base_coder import Coder
from aider.io import InputOutput


BENCH_HINT_DIRS: List[Path] = [
    Path("/benchmarks"),
    Path.cwd() / "tmp.benchmarks",              # /aider/tmp.benchmarks on host
    Path("/aider/tmp.benchmarks"),              # inside container if mounted
]


class ClaudeCodeCoder(Coder):
    """
    Minimal integration for @anthropic-ai/claude-code CLI as an Aider coder.

    Key points:
    - Uses Claude Code CLI directly (no base prompt formatting).
    - Detects the per-exercise *task root* and runs the CLI with cwd set there.
    - Hides test files during execution to prevent test feedback.
    - Leaves your aider command line & prompts unchanged.
    """
    edit_format = "claude-code"

    ALLOWED_TOOLS: List[str] = [
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

    def __init__(self, main_model, io: InputOutput, model_name: Optional[str] = None, **kwargs):
        super().__init__(main_model, io, **kwargs)
        self.conversational = True
        self.model_name = model_name  # If None, respect any ANTHROPIC_MODEL already in env.
        self._has_run = False  # ← Add this flag

        # Disable Aider retry windows to match Terminal Bench behavior
        for mod in ("aider.sendchat", "aider.coders.base_coder", "aider.models"):
            try:
                m = __import__(mod, fromlist=["RETRY_TIMEOUT"])
                setattr(m, "RETRY_TIMEOUT", None)
            except Exception:
                pass

        # Verify Claude Code CLI exists
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            ) from e

    # --------------------------------------------------------------------------------------------
    # CLI & env
    # --------------------------------------------------------------------------------------------
    def _build_env(self) -> Dict[str, str]:
        """
        Build a minimal env:
        - Require ANTHROPIC_API_KEY
        - Allow background tasks
        - Respect model (strip 'anthropic/' prefix for CLI)
        """
        env = {}

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        env["ANTHROPIC_API_KEY"] = api_key

        env["FORCE_AUTO_BACKGROUND_TASKS"] = "1"
        env["ENABLE_BACKGROUND_TASKS"] = "1"

        if self.model_name:
            env["ANTHROPIC_MODEL"] = 'claude-3-7-sonnet-20250219'
        elif "ANTHROPIC_MODEL" in os.environ:
            env["ANTHROPIC_MODEL"] = os.environ["ANTHROPIC_MODEL"]

        return env

    def _build_cmd(self, instruction: str) -> List[str]:
        return [
            "claude",
            "--verbose",
            "--output-format", "stream-json",
            "-p", instruction,
            "--allowedTools", " ".join(self.ALLOWED_TOOLS),
        ]

    def _run_claude_blocking(self, instruction: str, cwd: Optional[str]) -> Tuple[int, str, str]:
        cmd = self._build_cmd(instruction)
        env = self._build_env()
        run_cwd = cwd or os.getcwd()

        proc = subprocess.Popen(
            cmd,
            cwd=run_cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        out_buf: List[str] = []
        err_buf: List[str] = []

        # Stream stdout
        if proc.stdout is not None:
            for line in proc.stdout:
                if not line:
                    continue
                out_buf.append(line)
                self.io.tool_output(line.rstrip("\n"))

        # Read stderr at end
        if proc.stderr is not None:
            err_text = proc.stderr.read()
            if err_text:
                err_buf.append(err_text)
                self.io.tool_error(err_text.rstrip("\n"))

        exit_code = proc.wait()
        return exit_code, "".join(out_buf), "".join(err_buf)

    # --------------------------------------------------------------------------------------------
    # Snapshot/edit detection (so benchmark sees that we edited files)
    # --------------------------------------------------------------------------------------------
    def _snapshot_mtimes(self) -> Dict[str, float]:
        mtimes: Dict[str, float] = {}
        for fname in self.abs_fnames:
            try:
                mtimes[fname] = Path(fname).stat().st_mtime
            except FileNotFoundError:
                mtimes[fname] = 0.0
        return mtimes

    def _detect_edits(self, before: Dict[str, float]) -> bool:
        for fname, old_mtime in before.items():
            try:
                new_mtime = Path(fname).stat().st_mtime
            except FileNotFoundError:
                new_mtime = 0.0
            if new_mtime != old_mtime:
                return True
        return False

    # --------------------------------------------------------------------------------------------
    # Find the *exercise* root dir to use as cwd for the CLI
    # --------------------------------------------------------------------------------------------
    def _looks_like_task_root(self, root: Path) -> bool:
        # Java gradle or Exercism meta config both indicate a task root
        return (root / "build.gradle").exists() or (root / ".meta" / "config.json").exists()

    def _candidate_names(self) -> set[str]:
        # Try to infer solution filenames we care about
        names = {Path(fn).name for fn in (self.abs_fnames or [])}
        # Fallback common names seen in the suite (safe to include)
        if not names:
            names.update({"Alphametics.java", "AffineCipher.java", "BaseConverter.java"})
        return names

    def _walk_up_to_task_root(self, start: Path) -> Path | None:
        root = start
        for _ in range(10):
            if self._looks_like_task_root(root):
                return root
            if root.parent == root:
                break
            root = root.parent
        return None

    def _search_bench_for_match(self, names: set[str]) -> Path | None:
        """
        Scan known benchmark trees to find the current exercise dir that *contains*
        one of our solution filenames and also has build.gradle or .meta/config.json.
        Prefer the newest 'run' directory (by mtime) and any path that includes
        '/exercises/practice/'.
        """
        candidates: List[Tuple[float, Path]] = []

        def consider(p: Path):
            # Score by mtime and a small bonus for typical path layout
            mtime = p.stat().st_mtime if p.exists() else 0.0
            bonus = 10.0 if "/exercises/practice/" in p.as_posix() else 0.0
            candidates.append((mtime + bonus, p))

        for base in BENCH_HINT_DIRS:
            if not base.exists():
                continue

            # Look for meta configs first (fast)
            for meta in base.rglob(".meta/config.json"):
                task_root = meta.parent.parent  # .../<exercise>/.meta -> go up to <exercise>
                if not self._looks_like_task_root(task_root):
                    # Some tracks keep build.gradle above .meta
                    if not (task_root / "build.gradle").exists():
                        continue
                for name in names:
                    if list(task_root.rglob(name)):
                        consider(task_root)
                        break

            # Also accept plain gradle projects (some tracks)
            for gradle in base.rglob("build.gradle"):
                task_root = gradle.parent
                for name in names:
                    if list(task_root.rglob(name)):
                        consider(task_root)
                        break

        if not candidates:
            return None

        # Pick the most recent / best-scored candidate
        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]

    def _find_task_root(self) -> Path | None:
        """
        Robustly locate the current exercise directory:
          1) If any provided file path is *already inside* a task root, use that.
          2) Otherwise search benchmark folders (/benchmarks, tmp.benchmarks, etc.) for a task
             that contains the same solution filename(s).
          3) As a last resort, walk up from CWD for a build.gradle/.meta/config.json.
        """
        # 1) Try from known absolute file paths
        for fname in list(self.abs_fnames) if getattr(self, "abs_fnames", None) else []:
            p = Path(fname).resolve()
            root = self._walk_up_to_task_root(p.parent)
            if root:
                return root

        # 2) Search benchmark trees
        names = self._candidate_names()
        found = self._search_bench_for_match(names)
        if found:
            return found

        # 3) Walk up from current working directory
        root = self._walk_up_to_task_root(Path(os.getcwd()).resolve())
        return root

    # --------------------------------------------------------------------------------------------
    # Hide/restore test files to prevent agent access during development
    # --------------------------------------------------------------------------------------------
    def _load_task_config(self, task_root: Path) -> dict | None:
        """Load task config using aider's method for detecting test files"""
        config_file = task_root / ".meta" / "config.json"
        
        if not config_file.exists():
            return None
            
        try:
            with open(config_file) as f:
                config = json.loads(f.read())
            return config
        except Exception as e:
            print(f"[debug] Failed to load config from {config_file}: {e}")
            return None

    def _hide_test_files(self, task_root: Path, cfg: dict | None) -> dict[Path, Path]:
        """Temporarily move test files to hidden location to prevent agent access"""
        hidden_files = {}
        
        if not cfg:
            return hidden_files
        
        # Create hidden directory outside the workspace
        hidden_dir = task_root / ".hidden_tests_tmp"
        hidden_dir.mkdir(exist_ok=True)
        
        files = cfg.get("files", {})
        tests = files.get("test", []) or []
        examples = files.get("example", []) or []
        
        # Hide test files explicitly (most important)
        for rel in tests:
            original_path = task_root / rel
            if original_path.exists():
                hidden_path = hidden_dir / rel
                hidden_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Move file to hidden location
                shutil.move(str(original_path), str(hidden_path))
                hidden_files[original_path] = hidden_path
            else:
                print(f"[debug] Test file not found: {original_path}")

        # Hide example files explicitly  
        for rel in examples:
            original_path = task_root / rel
            if original_path.exists():
                hidden_path = hidden_dir / rel
                hidden_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Move file to hidden location
                shutil.move(str(original_path), str(hidden_path))
                hidden_files[original_path] = hidden_path
            else:
                print(f"[debug] Example file not found: {original_path}")

        # Hide .meta directory only (contains evaluation metadata)
        meta_dir = task_root / ".meta"
        if meta_dir.exists():
            hidden_path = hidden_dir / ".meta"
            
            try:
                # Use shutil.move for directories
                shutil.move(str(meta_dir), str(hidden_path))
                hidden_files[meta_dir] = hidden_path
            except Exception as e:
                print(f"[debug] Failed to hide .meta directory: {e}")

        return hidden_files

    def _restore_test_files(self, hidden_files: dict[Path, Path]):
        """Restore test files from hidden location"""
        for original_path, hidden_path in hidden_files.items():
            try:
                if hidden_path.exists():
                    original_path.parent.mkdir(parents=True, exist_ok=True)
                    hidden_path.rename(original_path)
            except Exception as e:
                print(f"[debug] Failed to restore {original_path}: {e}")
        
        # Clean up hidden directory
        if hidden_files:
            try:
                hidden_dir = list(hidden_files.values())[0].parent
                # Remove any remaining files/dirs in hidden directory
                for item in hidden_dir.rglob("*"):
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        item.rmdir()
                hidden_dir.rmdir()
            except Exception as e:
                print(f"[debug] Failed to clean up hidden directory: {e}")

    # --------------------------------------------------------------------------------------------
    # Entry points used by the benchmark harness
    # --------------------------------------------------------------------------------------------
    # We bypass the normal prompt-formatting pipeline entirely,
    def run(self, with_message: str, preproc: bool = False):
        """Return the CLI transcript text (what the harness expects), not self."""
        outs = self.run_one(with_message, preproc)
        return outs[0] if outs else ""

    def run_one(self, with_message, preproc: bool = False):
        response = self.send_new_user_message(with_message)
        return [response] if response is not None else []

    def send_message(self, message):
        response = self.send_new_user_message(message)
        return [response] if response is not None else []

    def format_messages(self):
        # Not used; we drive the CLI directly.
        return []

    def send_new_user_message(self, inp: str, files_content=None):
        if self._has_run:
            return ""
            
        self._has_run = True  # ← Mark as run
        before = self._snapshot_mtimes()

        task_root = self._find_task_root()
        hidden_files: dict[Path, Path] = {}
        
        if task_root:
            cfg = self._load_task_config(task_root)
            if cfg:
                hidden_files = self._hide_test_files(task_root, cfg)

        try:
            exit_code, out_text, _ = self._run_claude_blocking(
                inp, cwd=(str(task_root) if task_root else os.getcwd())
            )
        finally:
            if hidden_files:
                self._restore_test_files(hidden_files)

        self.edit_outcome = self._detect_edits(before)
        self.lint_outcome = None
        self.test_outcome = None

        if exit_code not in (0, None):
            raise RuntimeError(f"Claude command exited with {exit_code}")

        return out_text

    def send_test_failure_message(self, test_error_output: str, files_content=None):
        """Disabled to match Terminal Bench single-shot behavior."""
        return ""


__all__ = ["ClaudeCodeCoder"]