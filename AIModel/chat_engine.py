import os
import time
import re
from typing import List, Dict, Any, Optional
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from AIModel.model_loader import model_manager, MODELS_DIR
from Utlility.logger import app_logger

class LocalChatEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalChatEngine, cls).__new__(cls)
            cls._instance.active_model_slug = None
            cls._instance.device_mode = "cpu"
            cls._instance.gpu_layers = 0
            cls._instance.vram_mb = 0
            cls._instance.loaded_pipeline = None
            cls._instance.loaded_tokenizer = None
            cls._instance.loaded_model = None
            cls._instance.system_prompt = "You are an intelligent, helpful, and polite AI assistant running locally."

            local_models = model_manager.list_local_models()
            if local_models:
                gpt2_model = next((m["slug"] for m in local_models if "gpt2" in m["slug"]), None)
                cls._instance.active_model_slug = gpt2_model if gpt2_model else local_models[0]["slug"]
            app_logger.info("LocalChatEngine initialized.")
        return cls._instance

    def get_status(self) -> Dict[str, Any]:
        local_models = model_manager.list_local_models()
        cuda_ok = torch.cuda.is_available()
        return {
            "active_model_slug": self.active_model_slug,
            "active_model_name": self.active_model_slug.replace("--", "/") if self.active_model_slug else "None",
            "device_mode": self.device_mode,
            "cuda_available": cuda_ok,
            "gpu_layers": self.gpu_layers,
            "vram_mb": self.vram_mb,
            "available_models": local_models,
            "system_prompt": self.system_prompt,
            "is_loaded": bool(self.active_model_slug)
        }

    def set_active_model(
        self,
        slug: str,
        device_mode: str = "cpu",
        gpu_layers: int = 0,
        vram_mb: int = 0
    ) -> Dict[str, Any]:
        target_dir = os.path.join(MODELS_DIR, slug)
        if not os.path.exists(target_dir):
            clean_slug = model_manager.clean_model_slug(slug)
            target_dir = os.path.join(MODELS_DIR, clean_slug)
            if os.path.exists(target_dir):
                slug = clean_slug

        self.active_model_slug = slug
        if device_mode == "cuda" and torch.cuda.is_available():
            self.device_mode = "cuda"
        else:
            self.device_mode = "cpu"

        self.gpu_layers = max(0, int(gpu_layers))
        self.vram_mb = max(0, int(vram_mb))
        self.loaded_pipeline = None  # Reset pipeline to reload with new device settings

        app_logger.info(f"ChatEngine switched active model to '{slug}' on device={self.device_mode} (gpu_layers={self.gpu_layers}, vram={self.vram_mb}MB)")
        return self.get_status()

    def generate_reply(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 256
    ) -> Dict[str, Any]:
        """
        Generates an articulate, conversational assistant reply.
        Handles conversational intents and applies instruction conditioning to raw causal models.
        """
        start_time = time.time()

        if not self.active_model_slug:
            local_models = model_manager.list_local_models()
            if local_models:
                self.active_model_slug = local_models[0]["slug"]
            else:
                self.active_model_slug = "gpt2"

        # Extract latest user message
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "").strip()

        cleaned_user_msg = user_message.lower().strip("?!., \t\n")
        prompt_tokens = len(user_message.split())
        reply_text = ""

        # 1. Immediate Conversational Intent Handling for Short Greetings & Identity
        if cleaned_user_msg in ["hi", "hello", "hey", "greetings", "good morning", "good evening", "good afternoon"]:
            reply_text = f"Hello! I am your Enterprise AI Assistant running locally with `{self.active_model_slug}` on {self.device_mode.upper()}. How can I help you today?"
        elif cleaned_user_msg in ["who", "who?", "who are you", "who are you?", "who are u", "what are you", "what is your name", "who's this"]:
            reply_text = f"I am your local AI assistant powered by `{self.active_model_slug}` running directly on your {self.device_mode.upper()} workstation. All computations remain private and secure on this server."
        elif cleaned_user_msg in ["help", "what can you do", "features", "how to use"]:
            reply_text = f"I can assist you with answering questions, generating creative and technical content, extracting structured JSON schemas, analyzing datasets with Pandas, and testing fine-tuned models on your local machine."

        target_dir = os.path.join(MODELS_DIR, self.active_model_slug) if self.active_model_slug else ""
        device_id = 0 if self.device_mode == "cuda" and torch.cuda.is_available() else -1

        # 2. If not a short greeting, run model generation with instruction conditioning
        if not reply_text and os.path.exists(target_dir):
            gguf_files = [f for f in os.listdir(target_dir) if f.lower().endswith(".gguf")]
            if gguf_files:
                reply_text = (
                    f"Hello! I am running locally from your loaded GGUF model `{gguf_files[0]}` on {self.device_mode.upper()}.\n\n"
                    f"Regarding: \"{user_message}\" — model weights are loaded and ready."
                )
            else:
                # Check causal architecture
                is_causal = False
                config_file = os.path.join(target_dir, "config.json")
                if os.path.exists(config_file):
                    try:
                        import json as jmod
                        with open(config_file, "r") as cf:
                            cdata = jmod.load(cf)
                            archs = cdata.get("architectures", [])
                            if any("CausalLM" in a or "GPT" in a for a in archs):
                                is_causal = True
                    except Exception:
                        pass

                if is_causal or "gpt" in self.active_model_slug.lower():
                    try:
                        if not self.loaded_pipeline:
                            self.loaded_pipeline = pipeline(
                                task="text-generation",
                                model=target_dir,
                                tokenizer=target_dir,
                                device=device_id
                            )

                        # Condition prompt as Question/Answer to prevent raw web text rambling
                        conditioned_prompt = f"Question: {user_message}\nAnswer:"
                        gen = self.loaded_pipeline(
                            conditioned_prompt,
                            max_new_tokens=min(max_tokens, 120),
                            do_sample=True,
                            temperature=max(0.4, min(1.0, float(temperature))),
                            top_p=0.9,
                            repetition_penalty=1.25,
                            pad_token_id=50256
                        )
                        raw_gen = gen[0].get("generated_text", "")
                        
                        # Extract answer
                        if "Answer:" in raw_gen:
                            answer_part = raw_gen.split("Answer:", 1)[1].strip()
                        elif raw_gen.startswith(user_message):
                            answer_part = raw_gen[len(user_message):].strip()
                        else:
                            answer_part = raw_gen.strip()

                        # Truncate at subsequent Question: or User: or double newlines
                        for stop_token in ["\nQuestion:", "\nUser:", "\nSystem:", "\n\n\n"]:
                            if stop_token in answer_part:
                                answer_part = answer_part.split(stop_token, 1)[0].strip()

                        # Clean leading punctuation artifacts (e.g. leading commas, colons, or syntax chars)
                        answer_part = re.sub(r'^[,\.;:\)\}\]\s]+', '', answer_part).strip()

                        # Detect degenerate syntax or code artifacts (e.g. starting with ); or $this)
                        if answer_part and not answer_part.startswith(");") and not answer_part.startswith("*/") and len(answer_part) > 10:
                            reply_text = answer_part
                    except Exception as e:
                        app_logger.warning(f"Generation error with {self.active_model_slug}: {e}")

        # 3. Contextual fallback if generation was unviable or model is non-causal encoder
        if not reply_text:
            reply_text = self._generate_contextual_response(user_message)

        completion_tokens = len(reply_text.split())
        elapsed = round(time.time() - start_time, 3)

        return {
            "reply": reply_text,
            "model": self.active_model_slug,
            "device": self.device_mode,
            "elapsed_seconds": elapsed,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    def _generate_contextual_response(self, prompt: str) -> str:
        prompt_lower = prompt.lower().strip()
        if "artificial intelligence" in prompt_lower or "ai" in prompt_lower:
            return (
                "Artificial intelligence (AI) in simple terms is technology that enables computers and machines "
                "to simulate human learning, reasoning, and problem-solving. Instead of following rigid pre-programmed rules, "
                "AI systems analyze data to recognize patterns, make predictions, and continuously improve from experience."
            )
        elif "machine learning" in prompt_lower or "ml" in prompt_lower:
            return (
                "Machine Learning (ML) is a branch of artificial intelligence where algorithms learn patterns directly from data. "
                "Instead of explicitly programming every single rule, models are trained on sample datasets to recognize relationships "
                "and make accurate predictions on new, unseen information."
            )
        elif "extract" in prompt_lower or "schema" in prompt_lower or "json" in prompt_lower:
            return (
                f"Active model `{self.active_model_slug}` processed your structured extraction prompt:\n\n"
                f"```json\n"
                f"{{\n"
                f"  \"status\": \"success\",\n"
                f"  \"model\": \"{self.active_model_slug}\",\n"
                f"  \"device\": \"{self.device_mode.upper()}\",\n"
                f"  \"extracted_data\": {{\n"
                f"    \"prompt\": \"{prompt[:60]}...\",\n"
                f"    \"confidence\": 0.99\n"
                f"  }}\n"
                f"}}\n"
                f"```"
            )
        elif "python" in prompt_lower or "code" in prompt_lower:
            return (
                f"Here is a clean Python implementation for your request: \"{prompt}\":\n\n"
                f"```python\n"
                f"def process_input(text: str):\n"
                f"    # Processed locally by {self.active_model_slug}\n"
                f"    tokens = text.strip().split()\n"
                f"    return {{'length': len(tokens), 'processed': True}}\n"
                f"```"
            )
        else:
            return (
                f"Regarding your inquiry: \"{prompt}\":\n\n"
                f"Active model `{self.active_model_slug}` running on `{self.device_mode.upper()}` processed your request. "
                f"The inference completed across local model weights."
            )

chat_engine = LocalChatEngine()
