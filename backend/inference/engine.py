"""
DR. MIGI Inference Engine — Direct Model Execution
====================================================
Loads and runs the user's fine-tuned DR. MIGI model (suvidh21/dr-migi)
directly. No external API calls — the model runs in-process on CPU
(Streamlit Cloud) or GPU (local PC).
"""
from __future__ import annotations  # Python 3.9 compatibility for type hints

import os
import time
import torch
from threading import Thread
from transformers import TextIteratorStreamer
from backend.model.loader import load_model_and_tokenizer, determine_optimal_device, config
from backend.prompts.templates import build_chat_messages
from backend.utils.logger import logger, Timer


class DrMigiEngine:
    """
    The main execution orchestrator for DR. MIGI's reasoning brain.
    
    Loads the user's fine-tuned model (suvidh21/dr-migi) directly into memory
    and runs inference locally. On CPU (Streamlit Cloud), the model runs in
    float16 to fit within the 1 GB RAM limit. On local GPU, it runs in
    float16/bfloat16 for maximum speed.
    """
    def __init__(self, model_name: str | None = None):
        """
        Initializes the engine by loading the fine-tuned DR. MIGI model.
        
        On local PC (with GPU): Loads from disk (models/MIGI-Qwen2.5-0.5B-v1)
        On Streamlit Cloud (CPU): Downloads from suvidh21/dr-migi via HF_TOKEN
        """
        self.device, self.dtype = determine_optimal_device()
        self.model, self.tokenizer = load_model_and_tokenizer(model_name)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        logger.info(
            f"DrMigiEngine initialized on {self.device.upper()}. "
            f"Model ready for inference."
        )

    def generate_response(self, prompt: str, system_prompt: str | None = None, **generation_kwargs) -> dict:
        """
        Processes a prompt synchronously and returns the model response
        alongside performance metrics.
        
        Returns:
            dict with keys: response, input_tokens_count, output_tokens_count,
                           duration_seconds, tokens_per_second, device
        """
        params = config["default_generation_params"].copy()
        params.update(generation_kwargs)
        
        if system_prompt:
            messages = build_chat_messages(prompt, system_prompt)
        else:
            messages = build_chat_messages(prompt)
            
        start_time = time.perf_counter()
        
        # Apply chat template to format the conversation
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        
        # Move inputs to correct device (CPU for cloud, CUDA for local GPU)
        if self.device == "cuda":
            inputs = inputs.to(self.device)
            
        input_len = inputs["input_ids"].shape[1]
        
        logger.info(f"Prepared inputs with {input_len} prompt tokens. Generating...")
        
        # Generate response with the fine-tuned DR. MIGI model
        with torch.no_grad():
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
        Enables real-time streaming display in the Streamlit chat interface.
        """
        params = config["default_generation_params"].copy()
        params.update(generation_kwargs)
        
        if system_prompt:
            messages = build_chat_messages(prompt, system_prompt)
        else:
            messages = build_chat_messages(prompt)

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = inputs.to(self.device)
        
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
