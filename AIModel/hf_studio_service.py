import os
import sys
import io
import time
import json
import subprocess
import traceback
from typing import Dict, Any, List, Optional
import pandas as pd
from core.config import settings
from AIModel.model_loader import MODELS_DIR, model_manager
from Utlility.logger import app_logger

UPLOADED_DATASETS_DIR = os.path.join(os.path.dirname(MODELS_DIR), "datasets_upload")
os.makedirs(UPLOADED_DATASETS_DIR, exist_ok=True)

class HFStudioService:
    """Core backend engine for the unified Hugging Face Central Workstation."""

    def __init__(self):
        self.notebook_sessions: Dict[str, Dict[str, Any]] = {}

    # =========================================================================
    # 1. VISUAL DRAG & DROP & EXCEL / DATASET MANIPULATION
    # =========================================================================
    def parse_uploaded_dataset(self, filename: str, content: bytes) -> Dict[str, Any]:
        """
        Parses uploaded Excel (.xlsx, .xls), CSV, JSON, or Parquet datasets.
        Returns sheets, column list, row count, and first 100 sample rows.
        """
        file_path = os.path.join(UPLOADED_DATASETS_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(content)

        ext = os.path.splitext(filename)[1].lower()
        sheets = ["Default"]
        active_sheet = "Default"
        df = None

        try:
            if ext in [".xlsx", ".xls"]:
                excel_file = pd.ExcelFile(file_path)
                sheets = excel_file.sheet_names
                active_sheet = sheets[0] if sheets else "Sheet1"
                df = pd.read_excel(file_path, sheet_name=active_sheet)
            elif ext == ".csv":
                df = pd.read_csv(file_path)
            elif ext in [".json", ".jsonl"]:
                try:
                    df = pd.read_json(file_path)
                except Exception:
                    df = pd.read_json(file_path, lines=True)
            elif ext == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                # Text or unknown format fallback
                df = pd.read_csv(file_path, delimiter="\t")
        except Exception as e:
            app_logger.error(f"Error parsing dataset file {filename}: {e}")
            raise ValueError(f"Failed to parse file: {str(e)}")

        df = df.fillna("")
        columns = list(df.columns)
        total_rows = len(df)
        preview_rows = df.head(100).to_dict(orient="records")

        return {
            "filename": filename,
            "filepath": file_path,
            "file_size_kb": round(len(content) / 1024, 2),
            "sheets": sheets,
            "active_sheet": active_sheet,
            "columns": [str(c) for c in columns],
            "total_rows": total_rows,
            "preview_rows": preview_rows
        }

    def manipulate_dataset(
        self,
        filename: str,
        active_sheet: Optional[str] = None,
        rename_columns: Optional[Dict[str, str]] = None,
        drop_columns: Optional[List[str]] = None,
        filter_query: Optional[str] = None,
        new_sheet_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Performs in-memory and on-disk manipulation:
        Renaming columns, changing sheet names, filtering rows, and dropping fields.
        """
        file_path = os.path.join(UPLOADED_DATASETS_DIR, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file '{filename}' not found.")

        ext = os.path.splitext(filename)[1].lower()
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, sheet_name=active_sheet or 0)
        elif ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in [".json", ".jsonl"]:
            try:
                df = pd.read_json(file_path)
            except Exception:
                df = pd.read_json(file_path, lines=True)
        else:
            df = pd.read_parquet(file_path)

        # 1. Rename columns
        if rename_columns:
            df = df.rename(columns=rename_columns)

        # 2. Drop columns
        if drop_columns:
            valid_drops = [c for c in drop_columns if c in df.columns]
            df = df.drop(columns=valid_drops)

        # 3. Filter query
        if filter_query and filter_query.strip():
            try:
                df = df.query(filter_query)
            except Exception as e:
                app_logger.warning(f"Filter query error: {e}")

        # Save back manipulated file
        output_filename = f"manipulated_{filename}"
        output_path = os.path.join(UPLOADED_DATASETS_DIR, output_filename)
        
        if ext in [".xlsx", ".xls"]:
            sheet_title = new_sheet_name or active_sheet or "Sheet1"
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name=sheet_title, index=False)
        elif ext == ".csv":
            df.to_csv(output_path, index=False)
        else:
            df.to_json(output_path, orient="records", indent=2)

        df = df.fillna("")
        return {
            "success": True,
            "filename": output_filename,
            "filepath": output_path,
            "sheet_name": new_sheet_name or active_sheet or "Sheet1",
            "columns": list(df.columns),
            "total_rows": len(df),
            "preview_rows": df.head(100).to_dict(orient="records"),
            "message": f"Successfully updated dataset columns and schema."
        }

    # =========================================================================
    # 2. EMBEDDED HUGGING FACE CLI RUNNER
    # =========================================================================
    def run_hf_cli(self, command_str: str) -> Dict[str, Any]:
        """
        Executes a Hugging Face CLI command (e.g. 'hf auth whoami', 'hf cache scan', 'hf env').
        Uses the virtual environment's hf binary.
        """
        start_time = time.time()
        command_str = command_str.strip()

        # Sanitize command prefix
        clean_args = []
        if command_str.startswith("huggingface-cli"):
            parts = command_str.split()[1:]
            clean_args = [sys.executable, "-m", "huggingface_hub.cli"] + parts
        elif command_str.startswith("hf"):
            parts = command_str.split()[1:]
            # Use hf executable if present or python module
            hf_exe = os.path.join(os.path.dirname(sys.executable), "hf.exe")
            if os.path.exists(hf_exe):
                clean_args = [hf_exe] + parts
            else:
                clean_args = [sys.executable, "-m", "huggingface_hub.cli"] + parts
        else:
            clean_args = [sys.executable, "-m", "huggingface_hub.cli"] + command_str.split()

        env = os.environ.copy()
        if settings.HF_TOKEN:
            env["HF_TOKEN"] = settings.HF_TOKEN
            env["HUGGING_FACE_HUB_TOKEN"] = settings.HF_TOKEN

        try:
            proc = subprocess.run(
                clean_args,
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
                cwd=os.path.dirname(MODELS_DIR)
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            elapsed = round(time.time() - start_time, 3)

            return {
                "command": command_str,
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "elapsed_seconds": elapsed,
                "success": proc.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {
                "command": command_str,
                "exit_code": 124,
                "stdout": "",
                "stderr": "Command execution timed out after 30 seconds.",
                "elapsed_seconds": 30.0,
                "success": False
            }
        except Exception as e:
            return {
                "command": command_str,
                "exit_code": 1,
                "stdout": "",
                "stderr": str(e),
                "elapsed_seconds": round(time.time() - start_time, 3),
                "success": False
            }

    # =========================================================================
    # 3. INTERACTIVE AI / PYTORCH NOTEBOOK CELL EXECUTOR
    # =========================================================================
    def execute_notebook_cell(self, code: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Executes a Python code cell inside an interactive kernel session.
        Pre-injects torch, transformers, datasets, pandas, and hardware references.
        """
        if session_id not in self.notebook_sessions:
            import torch
            import transformers
            import datasets
            self.notebook_sessions[session_id] = {
                "globals": {
                    "torch": torch,
                    "transformers": transformers,
                    "datasets": datasets,
                    "pd": pd,
                    "os": os,
                    "time": time,
                    "json": json,
                    "MODELS_DIR": MODELS_DIR,
                    "device": "cuda" if torch.cuda.is_available() else "cpu"
                }
            }

        session_globals = self.notebook_sessions[session_id]["globals"]

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf

        start_time = time.time()
        result_repr = None
        error = None

        try:
            # First attempt eval for single expression, else exec
            try:
                compiled = compile(code, "<notebook-cell>", "eval")
                res = eval(compiled, session_globals)
                if res is not None:
                    result_repr = repr(res)
            except SyntaxError:
                compiled = compile(code, "<notebook-cell>", "exec")
                exec(compiled, session_globals)
        except Exception as e:
            error = traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        stdout_val = stdout_buf.getvalue()
        stderr_val = stderr_buf.getvalue()
        elapsed = round(time.time() - start_time, 3)

        return {
            "stdout": stdout_val,
            "stderr": stderr_val,
            "result": result_repr,
            "error": error,
            "elapsed_seconds": elapsed,
            "success": error is None
        }

hf_studio_service = HFStudioService()
