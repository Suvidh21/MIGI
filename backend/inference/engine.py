import os
import time
from threading import Thread
from transformers import TextIteratorStreamer
from backend.model.loader import load_model_and_tokenizer, determine_optimal_device, config
from backend.prompts.templates import build_chat_messages
from backend.utils.logger import logger, Timer

try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


def _resolve_hf_token() -> str | None:
    """Discovers Hugging Face access token from environment or Streamlit secrets."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    return None


class DrMigiEngine:
    """
    The main execution orchestrator for DR. MIGI's reasoning brain.
    
    Supports dual execution modes:
    1. Cloud Serverless Mode (Hugging Face Inference API):
       Used when HF_TOKEN is detected. Queries Hugging Face GPU servers (e.g. Qwen 2.5 72B)
       with zero local RAM footprint (<50MB RAM), preventing cloud memory crashes.
    2. Local PyTorch Mode:
       Used when running locally on PC with local weights / GPU.
    """
    def __init__(self, model_name: str | None = None, token: str | None = None):
        """
        Initializes the engine. Automatically detects if Hugging Face API mode
        should be activated based on available token.
        """
        resolved_token = token or _resolve_hf_token()
        
        if resolved_token and HF_HUB_AVAILABLE:
            self.use_api = True
            self.hf_token = resolved_token
            self.api_model = model_name or config.get("api_model_name", "Qwen/Qwen2.5-72B-Instruct")
            self.device = "Hugging Face Cloud GPU"
            self.model = None
            self.tokenizer = None
            self.client = InferenceClient(model=self.api_model, token=self.hf_token)
            logger.info(f"DrMigiEngine initialized in Cloud API mode with model '{self.api_model}'.")
        else:
            self.use_api = False
            self.client = None
            self.device, self.dtype = determine_optimal_device()
            self.model, self.tokenizer = load_model_and_tokenizer(model_name)
            
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"DrMigiEngine initialized in Local PyTorch mode on {self.device.upper()}.")

    def generate_response(self, prompt: str, system_prompt: str | None = None, **generation_kwargs) -> dict:
        """
        Processes a prompt synchronously and returns the model response alongside performance metrics.
        """
        params = config["default_generation_params"].copy()
        params.update(generation_kwargs)
        
        if system_prompt:
            messages = build_chat_messages(prompt, system_prompt)
        else:
            messages = build_chat_messages(prompt)
            
        start_time = time.perf_counter()
        
        # Branch 1: Cloud API Inference
        if self.use_api:
            res = self.client.chat.completions.create(
                messages=messages,
                max_tokens=params.get("max_new_tokens", 400),
                temperature=params.get("temperature", 0.3),
                top_p=params.get("top_p", 0.9),
            )
            duration = time.perf_counter() - start_time
            decoded_response = res.choices[0].message.content or ""
            
            # Extract or estimate token counts
            output_tokens = getattr(getattr(res, "usage", None), "completion_tokens", None)
            if output_tokens is None:
                output_tokens = len(decoded_response.split()) * 4 // 3
                
            input_tokens = getattr(getattr(res, "usage", None), "prompt_tokens", None)
            if input_tokens is None:
                input_tokens = len(prompt.split()) * 4 // 3
                
            tokens_per_second = output_tokens / duration if duration > 0 else 0.0
            
            metrics = {
                "response": decoded_response,
                "input_tokens_count": input_tokens,
                "output_tokens_count": output_tokens,
                "duration_seconds": duration,
                "tokens_per_second": tokens_per_second,
                "device": self.device
            }
            logger.info(
                f"Cloud API generated {output_tokens} tokens in {duration:.2f}s "
                f"({tokens_per_second:.2f} tok/s) via {self.api_model}."
            )
            return metrics

        # Branch 2: Local PyTorch Inference
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]
        
        logger.info(f"Prepared inputs with {input_len} prompt tokens. Initiating generation...")
        
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=params.get("max_new_tokens", 256),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            top_k=params.get("top_k", 50),
            do_sample=params.get("do_sample", True),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        duration = time.perf_counter() - start_time
        generated_tokens = output_ids[0][input_len:]
        num_generated_tokens = len(generated_tokens)
        decoded_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        tokens_per_second = num_generated_tokens / duration if duration > 0 else 0.0
        
        metrics = {
            "response": decoded_response,
            "input_tokens_count": input_len,
            "output_tokens_count": num_generated_tokens,
            "duration_seconds": duration,
            "tokens_per_second": tokens_per_second,
            "device": self.device
        }
        logger.info(
            f"Generated {num_generated_tokens} tokens in {duration:.2f}s "
            f"({tokens_per_second:.2f} tok/s) on {self.device.upper()}."
        )
        return metrics

    def generate_stream(self, prompt: str, system_prompt: str | None = None, **generation_kwargs):
        """
        Yields tokens one by one as they are produced by the LLM.
        """
        params = config["default_generation_params"].copy()
        params.update(generation_kwargs)
        
        if system_prompt:
            messages = build_chat_messages(prompt, system_prompt)
        else:
            messages = build_chat_messages(prompt)
            
        if self.use_api:
            stream = self.client.chat.completions.create(
                messages=messages,
                max_tokens=params.get("max_new_tokens", 400),
                temperature=params.get("temperature", 0.3),
                top_p=params.get("top_p", 0.9),
                stream=True
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            return

        # Local PyTorch streamer
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_args = dict(
            streamer=streamer,
            max_new_tokens=params.get("max_new_tokens", 256),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            top_k=params.get("top_k", 50),
            do_sample=params.get("do_sample", True),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        generation_kwargs_full = dict(**inputs, **generation_args)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs_full)
        thread.start()
        
        for new_text in streamer:
            yield new_text
            
        thread.join()

