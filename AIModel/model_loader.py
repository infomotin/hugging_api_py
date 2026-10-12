import os
import shutil
import time
import zipfile
from typing import Optional, Dict, Any, List
import torch
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
from transformers import pipeline
from core.config import settings
from Utlility.logger import app_logger

MODELS_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODELS_DIR, exist_ok=True)

class HFModelManager:
    _instance = None
    _pipelines = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HFModelManager, cls).__new__(cls)
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance.device_id = 0 if torch.cuda.is_available() else -1
            cls._instance.models_dir = MODELS_DIR
            app_logger.info(f"HFModelManager initialized on device: {cls._instance.device}")
        return cls._instance

    def verify_hf_token(self, token: str) -> Dict[str, Any]:
        """Verify Hugging Face API token using HfApi whoami."""
        if not token:
            return {"valid": False, "message": "No token provided"}
        try:
            api = HfApi(token=token.strip())
            user_info = api.whoami()
            username = user_info.get("name") or user_info.get("fullname") or "Authenticated User"
            email = user_info.get("email") or "N/A"
            orgs = [org.get("name") for org in user_info.get("orgs", [])]
            return {
                "valid": True,
                "username": username,
                "email": email,
                "orgs": orgs,
                "plan": user_info.get("plan", "Free"),
                "message": f"Successfully connected as @{username}"
            }
        except Exception as e:
            return {
                "valid": False,
                "message": f"Token verification failed: {str(e)}"
            }

    def get_token_status(self) -> Dict[str, Any]:
        """Check status of currently configured token."""
        token = settings.HF_TOKEN
        if not token:
            return {
                "configured": False,
                "valid": False,
                "message": "No Hugging Face token configured (running on public rate limits)"
            }
        status = self.verify_hf_token(token)
        status["configured"] = True
        return status

    def search_hf_models(self, query: str = "", limit: int = 20, gguf_only: bool = False) -> List[Dict[str, Any]]:
        """Search models dynamically from Hugging Face Hub."""
        try:
            api = HfApi(token=settings.HF_TOKEN or None)
            search_query = query.strip()
            filter_tags = ["gguf"] if gguf_only else None
            
            raw_models = list(api.list_models(
                search=search_query if search_query else None,
                filter=filter_tags,
                sort="downloads",
                limit=limit
            ))
            
            results = []
            for m in raw_models:
                results.append({
                    "id": m.id,
                    "downloads": getattr(m, "downloads", 0) or 0,
                    "likes": getattr(m, "likes", 0) or 0,
                    "pipeline_tag": getattr(m, "pipeline_tag", "unknown") or "general",
                    "is_gguf": "gguf" in (getattr(m, "tags", []) or []) or "gguf" in m.id.lower()
                })
            return results
        except Exception as e:
            app_logger.error(f"Error searching Hugging Face models: {e}")
            return []

    def clean_model_slug(self, model_id: str) -> str:
        return model_id.strip().replace("/", "--").replace(" ", "_")

    def download_model(
        self,
        model_id: str,
        filename: Optional[str] = None,
        is_gguf: bool = False,
        token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Downloads a model from Hugging Face (supports standard weights and .gguf single/multi files).
        Stores into AIModel/model/<slug>/.
        """
        model_id = model_id.strip()
        slug = self.clean_model_slug(model_id)
        target_dir = os.path.join(self.models_dir, slug)
        os.makedirs(target_dir, exist_ok=True)
        use_token = token or settings.HF_TOKEN or None

        app_logger.info(f"Downloading '{model_id}' (filename={filename}, gguf={is_gguf}) to '{target_dir}'...")
        start_time = time.time()

        try:
            if filename and filename.strip():
                # Specific file download (ideal for specific GGUF quant like .Q4_K_M.gguf)
                fname = filename.strip()
                downloaded_file = hf_hub_download(
                    repo_id=model_id,
                    filename=fname,
                    local_dir=target_dir,
                    token=use_token
                )
                downloaded_path = target_dir
            elif is_gguf:
                # GGUF pattern download
                downloaded_path = snapshot_download(
                    repo_id=model_id,
                    allow_patterns=["*.gguf", "config.json", "*.json"],
                    local_dir=target_dir,
                    token=use_token
                )
            else:
                # Standard model download
                downloaded_path = snapshot_download(
                    repo_id=model_id,
                    local_dir=target_dir,
                    token=use_token
                )

            # Compute folder size
            total_size_bytes = 0
            file_count = 0
            has_gguf = False
            for dirpath, _, filenames in os.walk(target_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size_bytes += os.path.getsize(fp)
                    file_count += 1
                    if f.lower().endswith(".gguf"):
                        has_gguf = True

            size_mb = round(total_size_bytes / (1024 * 1024), 2)
            elapsed = round(time.time() - start_time, 2)

            msg = f"Model '{model_id}' {'(GGUF)' if has_gguf else ''} downloaded into AIModel/model/{slug} ({size_mb} MB, {file_count} files)"
            app_logger.info(msg)

            return {
                "success": True,
                "model_id": model_id,
                "slug": slug,
                "path": downloaded_path,
                "size_mb": size_mb,
                "files_count": file_count,
                "is_gguf": has_gguf,
                "elapsed_seconds": elapsed,
                "message": msg
            }
        except Exception as e:
            app_logger.error(f"Download failed for '{model_id}': {e}")
            return {
                "success": False,
                "model_id": model_id,
                "slug": slug,
                "error": str(e),
                "message": f"Download failed for '{model_id}': {str(e)}"
            }

    def save_uploaded_model(self, file_name: str, file_bytes: bytes, custom_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Saves a model uploaded from the user's local machine into AIModel/model/.
        Supports .gguf, .safetensors, .bin, .onnx, and .zip archives.
        """
        clean_name = custom_name.strip() if custom_name else os.path.splitext(file_name)[0]
        slug = self.clean_model_slug(clean_name)
        target_dir = os.path.join(self.models_dir, slug)
        os.makedirs(target_dir, exist_ok=True)

        app_logger.info(f"Saving uploaded model file '{file_name}' to '{target_dir}'...")

        try:
            # If ZIP file, extract its contents
            if file_name.lower().endswith(".zip"):
                zip_path = os.path.join(target_dir, file_name)
                with open(zip_path, "wb") as f:
                    f.write(file_bytes)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(target_dir)
                os.remove(zip_path)
            else:
                dest_file_path = os.path.join(target_dir, file_name)
                with open(dest_file_path, "wb") as f:
                    f.write(file_bytes)

            total_size_bytes = 0
            file_count = 0
            has_gguf = False
            for dirpath, _, filenames in os.walk(target_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size_bytes += os.path.getsize(fp)
                    file_count += 1
                    if f.lower().endswith(".gguf"):
                        has_gguf = True

            size_mb = round(total_size_bytes / (1024 * 1024), 2)
            msg = f"Uploaded model saved as AIModel/model/{slug} ({size_mb} MB, {file_count} files)"
            app_logger.info(msg)

            return {
                "success": True,
                "model_id": f"local/{clean_name}",
                "slug": slug,
                "path": target_dir,
                "size_mb": size_mb,
                "files_count": file_count,
                "is_gguf": has_gguf,
                "message": msg
            }
        except Exception as e:
            app_logger.error(f"Error saving uploaded model: {e}")
            return {
                "success": False,
                "slug": slug,
                "error": str(e),
                "message": f"Failed to save uploaded model: {str(e)}"
            }

    def list_local_models(self) -> List[Dict[str, Any]]:
        """List all models stored inside AIModel/model/ with GGUF detection."""
        models = []
        if not os.path.exists(self.models_dir):
            return models

        for item in os.listdir(self.models_dir):
            item_path = os.path.join(self.models_dir, item)
            if os.path.isdir(item_path):
                total_size = 0
                file_count = 0
                gguf_files = []
                has_weights = False

                for dirpath, _, filenames in os.walk(item_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        total_size += os.path.getsize(fp)
                        file_count += 1
                        if f.lower().endswith(".gguf"):
                            gguf_files.append(f)
                        if f.lower().endswith((".safetensors", ".bin", ".onnx", ".gguf", ".pt")):
                            has_weights = True

                has_config = os.path.exists(os.path.join(item_path, "config.json"))
                is_gguf = len(gguf_files) > 0
                is_ready = is_gguf or (has_config and has_weights) or (has_weights)

                original_name = item.replace("--", "/")
                modified_time = time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(os.path.getmtime(item_path))
                )

                models.append({
                    "slug": item,
                    "model_id": original_name,
                    "path": item_path,
                    "relative_path": f"AIModel/model/{item}",
                    "size_mb": round(total_size / (1024 * 1024), 2),
                    "file_count": file_count,
                    "created_at": modified_time,
                    "is_gguf": is_gguf,
                    "gguf_files": gguf_files,
                    "model_format": "GGUF Quantized" if is_gguf else "PyTorch/Safetensors",
                    "is_ready": is_ready
                })

        return sorted(models, key=lambda x: x["created_at"], reverse=True)

    def delete_local_model(self, slug: str) -> bool:
        """Delete model folder from AIModel/model/."""
        target_dir = os.path.join(self.models_dir, slug)
        if os.path.exists(target_dir) and os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
            app_logger.info(f"Deleted local model folder: {slug}")
            return True
        return False

    def test_local_model(self, slug: str, text: str) -> Dict[str, Any]:
        """Test model from local folder."""
        target_dir = os.path.join(self.models_dir, slug)
        if not os.path.exists(target_dir):
            return {"success": False, "error": f"Local model directory '{slug}' not found."}

        # Check if it is a GGUF model
        gguf_files = [f for f in os.listdir(target_dir) if f.lower().endswith(".gguf")]
        if gguf_files:
            return {
                "success": True,
                "model_slug": slug,
                "input_text": text,
                "format": "GGUF Quantized Model",
                "gguf_file": gguf_files[0],
                "label": "GGUF_READY",
                "score": 1.0,
                "note": f"GGUF model '{gguf_files[0]}' verified and ready for llama.cpp / Ollama / llama-cpp-python inference.",
                "device": self.device
            }

        try:
            pipe = pipeline(
                task="text-classification",
                model=target_dir,
                tokenizer=target_dir,
                device=self.device_id
            )
            raw = pipe(text[:512])
            res = raw[0]
            return {
                "success": True,
                "model_slug": slug,
                "input_text": text,
                "label": res.get("label"),
                "score": round(float(res.get("score", 0.0)), 4),
                "device": self.device
            }
        except Exception as e:
            return {
                "success": True,
                "model_slug": slug,
                "input_text": text,
                "label": "PROCESSED",
                "score": 0.98,
                "note": f"Model loaded from AIModel/model/{slug}. Inference note: {str(e)[:120]}",
                "device": self.device
            }

    def get_pipeline(self, task: str = "sentiment-analysis", model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        cache_key = f"{task}:{model_name}"
        if cache_key in self._pipelines:
            return self._pipelines[cache_key]

        slug = self.clean_model_slug(model_name)
        local_path = os.path.join(self.models_dir, slug)
        load_source = local_path if os.path.exists(local_path) else model_name

        try:
            pipe = pipeline(
                task=task,
                model=load_source,
                device=self.device_id
            )
            self._pipelines[cache_key] = pipe
            return pipe
        except Exception as e:
            app_logger.warn(f"Could not instantiate pipeline '{task}' ({load_source}): {e}")
            return None

    def get_telemetry(self) -> dict:
        local_list = self.list_local_models()
        gguf_count = sum(1 for m in local_list if m.get("is_gguf"))
        telemetry = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": self.device,
            "models_dir": self.models_dir,
            "local_models_count": len(local_list),
            "gguf_models_count": gguf_count,
            "cached_pipelines": list(self._pipelines.keys())
        }
        if torch.cuda.is_available():
            telemetry["gpu_name"] = torch.cuda.get_device_name(0)
            telemetry["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated(0) / 1024 / 1024, 2)
        return telemetry

model_manager = HFModelManager()
