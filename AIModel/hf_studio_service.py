import os
import sys
import io
import time
import json
import shutil
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
    """Enterprise backend engine for the Hugging Face Central Workstation."""

    def __init__(self):
        self.notebook_sessions: Dict[str, Dict[str, Any]] = {}
        self.workspace_dir = os.path.abspath(os.path.dirname(MODELS_DIR))

    # =========================================================================
    # 1. VISUAL DRAG & DROP & EXCEL / DATASET MANIPULATION
    # =========================================================================
    def list_uploaded_datasets(self) -> List[Dict[str, Any]]:
        """Lists all uploaded datasets in the datasets_upload directory with file metadata."""
        items = []
        if not os.path.exists(UPLOADED_DATASETS_DIR):
            return items

        for f in os.listdir(UPLOADED_DATASETS_DIR):
            fp = os.path.join(UPLOADED_DATASETS_DIR, f)
            if os.path.isfile(fp):
                stat = os.stat(fp)
                ext = os.path.splitext(f)[1].lower()
                items.append({
                    "filename": f,
                    "filepath": fp,
                    "file_format": ext.replace(".", "").upper(),
                    "size_kb": round(stat.st_size / 1024, 2),
                    "modified_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
                })
        # Sort newest first
        items.sort(key=lambda x: x["modified_at"], reverse=True)
        return items

    def parse_uploaded_dataset(self, filename: str, content: bytes) -> Dict[str, Any]:
        """
        Parses uploaded Excel (.xlsx, .xls), CSV, JSON, or Parquet datasets.
        Returns sheets, column list, row count, and preview rows.
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
                df = pd.read_csv(file_path, delimiter="\t")
        except Exception as e:
            app_logger.error(f"Error parsing dataset file {filename}: {e}")
            raise ValueError(f"Failed to parse file: {str(e)}")

        df = df.fillna("")
        columns = list(df.columns)
        total_rows = len(df)
        preview_rows = df.head(100).to_dict(orient="records")

        # Column profile stats
        profiles = []
        for c in columns[:25]:
            profiles.append({
                "column": str(c),
                "dtype": str(df[c].dtype),
                "null_count": int(df[c].isnull().sum()) if hasattr(df[c], "isnull") else 0,
                "unique_count": int(df[c].nunique()) if hasattr(df[c], "nunique") else 0
            })

        return {
            "filename": filename,
            "filepath": file_path,
            "file_size_kb": round(len(content) / 1024, 2),
            "sheets": sheets,
            "active_sheet": active_sheet,
            "columns": [str(c) for c in columns],
            "column_profiles": profiles,
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
        """Performs in-memory and on-disk dataset schema transformations and export."""
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
        base_name = os.path.splitext(filename)[0]
        output_filename = f"transformed_{base_name}{ext}"
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
            "columns": [str(c) for c in df.columns],
            "total_rows": len(df),
            "preview_rows": df.head(100).to_dict(orient="records"),
            "message": f"Successfully transformed and saved dataset as '{output_filename}'."
        }

    def delete_dataset(self, filename: str) -> bool:
        """Removes a dataset file from storage."""
        file_path = os.path.join(UPLOADED_DATASETS_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False

    # =========================================================================
    # 2. REAL PROMPT & COM / TERMINAL CLI ENGINE
    # =========================================================================
    def run_hf_cli(self, command_str: str) -> Dict[str, Any]:
        """
        Executes interactive commands in COM/Terminal mode:
        Supports Hugging Face CLI ('hf auth whoami', 'hf cache scan', 'hf env'),
        Python/Pip commands ('python -V', 'pip list'), and shell commands.
        """
        start_time = time.time()
        command_str = command_str.strip()
        prompt_prefix = f"HF-STUDIO {self.workspace_dir}>"

        if not command_str:
            return {
                "command": "",
                "prompt_prefix": prompt_prefix,
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "elapsed_seconds": 0.0,
                "success": True
            }

        # Handle clear screen command
        if command_str.lower() in ["cls", "clear"]:
            return {
                "command": command_str,
                "prompt_prefix": prompt_prefix,
                "clear": True,
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "elapsed_seconds": 0.0,
                "success": True
            }

        env = os.environ.copy()
        if settings.HF_TOKEN:
            env["HF_TOKEN"] = settings.HF_TOKEN
            env["HUGGING_FACE_HUB_TOKEN"] = settings.HF_TOKEN

        # Dispatch command
        is_shell = False
        parts = command_str.split()
        first_token = parts[0].lower()

        if first_token in ["hf", "huggingface-cli"]:
            sub_args = parts[1:]
            hf_exe = os.path.join(os.path.dirname(sys.executable), "hf.exe")
            if os.path.exists(hf_exe):
                args = [hf_exe] + sub_args
            else:
                args = [sys.executable, "-m", "huggingface_hub.cli"] + sub_args
        elif first_token in ["python", "py"]:
            args = [sys.executable] + parts[1:]
        elif first_token == "pip":
            args = [sys.executable, "-m", "pip"] + parts[1:]
        else:
            # General shell execution on Windows
            args = command_str
            is_shell = True

        try:
            proc = subprocess.run(
                args,
                shell=is_shell,
                capture_output=True,
                text=True,
                timeout=45,
                env=env,
                cwd=self.workspace_dir
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            elapsed = round(time.time() - start_time, 3)

            return {
                "command": command_str,
                "prompt_prefix": prompt_prefix,
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "elapsed_seconds": elapsed,
                "success": proc.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {
                "command": command_str,
                "prompt_prefix": prompt_prefix,
                "exit_code": 124,
                "stdout": "",
                "stderr": f"Command timed out after 45 seconds.",
                "elapsed_seconds": 45.0,
                "success": False
            }
        except Exception as e:
            return {
                "command": command_str,
                "prompt_prefix": prompt_prefix,
                "exit_code": 1,
                "stdout": "",
                "stderr": str(e),
                "elapsed_seconds": round(time.time() - start_time, 3),
                "success": False
            }

    # =========================================================================
    # 3. INTERACTIVE AI / PYTORCH MULTI-CELL NOTEBOOK EXECUTOR
    # =========================================================================
    def execute_notebook_cell(self, code: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Executes a Python code cell inside an interactive stateful kernel session.
        Pre-injects torch, transformers, datasets, pandas, and hardware diagnostic tools.
        Automatically formats DataFrame expressions into styled HTML tables.
        """
        if session_id not in self.notebook_sessions:
            import torch
            import transformers
            import datasets

            def profile_model(model_name: str):
                """Helper function inside notebook to profile local model parameters."""
                slug = model_manager.clean_model_slug(model_name)
                m_path = os.path.join(MODELS_DIR, slug)
                if not os.path.exists(m_path):
                    print(f"Model '{model_name}' not found in AIModel/model/.")
                    return
                print(f"=== Model Architecture Profile: {model_name} ===")
                try:
                    from transformers import AutoModel
                    m = AutoModel.from_pretrained(m_path)
                    total_p = sum(p.numel() for p in m.parameters())
                    trainable_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
                    print(f"Total Parameters:     {total_p:,}")
                    print(f"Trainable Parameters: {trainable_p:,}")
                    print(f"Device Placement:     {next(m.parameters()).device}")
                except Exception as ex:
                    print(f"Profile error: {ex}")

            self.notebook_sessions[session_id] = {
                "globals": {
                    "torch": torch,
                    "transformers": transformers,
                    "datasets": datasets,
                    "pd": pd,
                    "os": os,
                    "sys": sys,
                    "time": time,
                    "json": json,
                    "MODELS_DIR": MODELS_DIR,
                    "model_manager": model_manager,
                    "profile_model": profile_model,
                    "device": "cuda" if torch.cuda.is_available() else "cpu"
                },
                "execution_count": 0
            }

        session = self.notebook_sessions[session_id]
        session["execution_count"] += 1
        session_globals = session["globals"]

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf

        start_time = time.time()
        result_repr = None
        html_table = None
        error = None

        import ast

        try:
            parsed_ast = ast.parse(code)
            if parsed_ast.body and isinstance(parsed_ast.body[-1], ast.Expr):
                last_expr = ast.Expression(parsed_ast.body[-1].value)
                parsed_ast.body.pop()
                if parsed_ast.body:
                    exec(compile(parsed_ast, "<notebook-cell>", "exec"), session_globals)
                res = eval(compile(last_expr, "<notebook-cell>", "eval"), session_globals)
                if res is not None:
                    if isinstance(res, pd.DataFrame):
                        html_table = res.head(25).to_html(classes="table table-sm", border=0)
                        result_repr = f"DataFrame: {res.shape[0]} rows × {res.shape[1]} columns"
                    elif isinstance(res, pd.Series):
                        html_table = res.to_frame().head(25).to_html(classes="table table-sm", border=0)
                        result_repr = f"Series (length {len(res)})"
                    else:
                        result_repr = repr(res)
            else:
                exec(compile(parsed_ast, "<notebook-cell>", "exec"), session_globals)
        except Exception as e:
            error = traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        stdout_val = stdout_buf.getvalue()
        stderr_val = stderr_buf.getvalue()
        elapsed = round(time.time() - start_time, 3)

        return {
            "execution_count": session["execution_count"],
            "stdout": stdout_val,
            "stderr": stderr_val,
            "result": result_repr,
            "html_table": html_table,
            "error": error,
            "elapsed_seconds": elapsed,
            "success": error is None
        }

    def reset_notebook_session(self, session_id: str = "default") -> bool:
        """Resets the state of a notebook session kernel."""
        if session_id in self.notebook_sessions:
            del self.notebook_sessions[session_id]
        return True

hf_studio_service = HFStudioService()
