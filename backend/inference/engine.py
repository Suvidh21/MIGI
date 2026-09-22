import os
import time
from threading import Thread
from transformers import TextIteratorStreamer
from backend.model.loader import load_model_and_tokenizer, determine_optimal_device, config
from backend.prompts.templates import build_chat_messages
from backend.utils.logger import logger, Timer


class DrMigiEngine:
    """
    The main execution orchestrator for DR. MIGI's reasoning brain.
    
    Loads the user's fine-tuned model (suvidh21/dr-migi) directly into memory
    and runs inference locally. On CPU (Streamlit Cloud), the model is INT8
    quantized to fit within the 1 GB RAM limit. On local GPU, it runs in
    float16/bfloat16 for maximum speed.
    
    Architecture:
        [Private HF Hub: suvidh21/dr-migi] → (HF_TOKEN auth download) →
        [Local/Cloud CPU/GPU RAM] → (INT8 quantized on CPU) →
        [Live inference via generate()]
    """
    def __init__(self, model_name: str | None = None):
        """
        Initializes the engine by loading the fine-tuned DR. MIGI model
        and tokenizer into memory.
        
        On your local PC (with GPU): Loads from local disk (models/MIGI-Qwen2.5-0.5B-v1)
        On Streamlit Cloud (CPU only): Downloads from suvidh21/dr-migi via HF_TOKEN,
                                       then applies INT8 quantization to fit in 1 GB RAM.
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
        Processes a prompt synchronously and returns the model response alongside performance metrics.
        
        Arguments:
            prompt: The user's clinical question or message.
            system_prompt: Optional system-level instructions for the model.
            **generation_kwargs: Override default generation parameters (max_new_tokens, temperature, etc.)
            
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
        
        # Move inputs to correct device (CPU for quantized model, CUDA for GPU)
        if self.device == "cuda":
            inputs = inputs.to(self.device)
            
        input_len = inputs["input_ids"].shape[1]
        
        logger.info(f"Prepared inputs with {input_len} prompt tokens. Initiating generation...")
        
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

        # Local PyTorch streamer
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        
        # Move inputs to correct device
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


# Required import for torch.no_grad context manager
import torch
