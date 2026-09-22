import time
from threading import Thread
from transformers import TextIteratorStreamer
from backend.model.loader import load_model_and_tokenizer, determine_optimal_device, config
from backend.prompts.templates import build_chat_messages
from backend.utils.logger import logger, Timer

class DrMigiEngine:
    """
    The main execution orchestrator for DR. MIGI's reasoning brain.
    
    What it does:
        Encapsulates the tokenizer and causal language model. Prepares textual queries
        into format-compliant chat templates, feeds them to PyTorch, and handles decoding
        and generation performance logging.
        
    Why it is needed:
        Abstracts Hugging Face's raw API behind a clean, testable interface. Ensures
        uniform chat framing, temperature parameters, and resource logging across the app.
        
    Best practices:
        Avoid hardcoded configurations; fall back to loaded JSON configurations. Track
        generative metrics (Tokens per Second) to verify performance.
    """
    def __init__(self, model_name: str | None = None):
        """
        Initializes the engine, loading the model and tokenizer to the optimal device.
        """
        self.device, self.dtype = determine_optimal_device()
        self.model, self.tokenizer = load_model_and_tokenizer(model_name)
        
        # Verify if pad token is set (needed to suppress HF warning logs during batch execution)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate_response(self, prompt: str, system_prompt: str | None = None, **generation_kwargs) -> dict:
        """
        Processes a prompt synchronously and returns the model response alongside performance metrics.
        
        Arguments:
            prompt: User message content.
            system_prompt: Optional override for the default system persona.
            generation_kwargs: Overrides for default model hyperparameters (e.g. temperature, max_new_tokens).
            
        Returns:
            Dictionary containing response text, token counts, and execution metrics.
        """
        # Resolve generation hyperparameters: prioritize custom arguments, fallback to default config
        params = config["default_generation_params"].copy()
        params.update(generation_kwargs)
        
        # Build structured system & user message dictionaries
        if system_prompt:
            messages = build_chat_messages(prompt, system_prompt)
        else:
            messages = build_chat_messages(prompt)
            
        # Format the system/user block into Qwen special tokens syntax and tokenize it
        # What it does: apply_chat_template maps structured messages to the raw prompt sequence
        # (e.g., adding <|im_start|>system...<|im_end|>\n<|im_start|>user...)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]
        
        logger.info(f"Prepared inputs with {input_len} prompt tokens. Initiating generation...")
        
        # Autoregressive Generation
        start_time = time.perf_counter()
        
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
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Slice away the prefill prompt tokens from generated output
        generated_tokens = output_ids[0][input_len:]
        num_generated_tokens = len(generated_tokens)
        
        # Decode token IDs back into string characters
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
        
        Why it is needed:
            LLM generation on CPU is slow (high latency). Streaming returns text chunks
            immediately as they are calculated, improving the interactive feel of the companion.
            
        How it works:
            Initializes a TextIteratorStreamer, puts the model generation on a background
            Thread, and yields strings from the streamer queue on the main thread.
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
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        
        # TextIteratorStreamer ignores the prompt tokens if skip_prompt=True
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
        
        # Spin up generation in a background thread to prevent blocking
        # We pass kwargs containing **inputs and other generation args
        generation_kwargs_full = dict(**inputs, **generation_args)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs_full)
        thread.start()
        
        # Yield tokens as they arrive in the queue
        for new_text in streamer:
            yield new_text
            
        thread.join()
