import os
import sys

# Redirection setup: Ensure model cache is on D: drive to prevent filling C: drive.
os.environ["HF_HOME"] = "D:/Suvidh/Suvidh/DR.MIGI/models/cache"

# Append project root to path to allow importing modules from src directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model.loader import load_model_and_tokenizer, determine_optimal_device
from src.prompts.templates import build_chat_messages
from src.utils.logger import logger, Timer

def main():
    print("=====================================================================")
    print("DR. MIGI - PHASE 1: TOKENIZATION & GENERATION DEEP-DIVE")
    print("=====================================================================\n")

    # 1. Load model and tokenizer
    # Task 7: Load Qwen locally
    device, dtype = determine_optimal_device()
    model, tokenizer = load_model_and_tokenizer()
    
    print("\n---------------------------------------------------------------------")
    print("TASK 8: Running the first prompt through tokenization")
    print("---------------------------------------------------------------------")
    
    # Define a clean medical inquiry
    prompt = "What is the function of the heart?"
    messages = build_chat_messages(prompt)
    
    # Apply Chat Template (converts list of system/user messages to Qwen raw template text)
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("\n[FORMATTED PROMPT (Actual sequence seen by the model)]:")
    print(repr(formatted_text)) # Use repr to show hidden characters like newlines
    
    # Tokenize input text to get token IDs
    inputs = tokenizer(formatted_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    print(f"\n[INPUT TOKEN IDs TENSOR ({input_ids.shape[0]} batch size, {input_ids.shape[1]} sequence length)]:")
    print(input_ids)
    
    print("\n[STEP-BY-STEP INPUT DECODING (Subwords mapping)]:")
    # We loop over every token ID in our prompt and decode it individually.
    # This demonstrates BPE (Byte-Pair Encoding) partitioning.
    for i, token_id in enumerate(input_ids[0].tolist()):
        decoded_string = tokenizer.decode([token_id])
        print(f"  Pos {i:3d} | Token ID: {token_id:6d} | Subword: {repr(decoded_string)}")
        
    print("\n---------------------------------------------------------------------")
    print("TASK 9: Generation & Explanation of every single generated token")
    print("---------------------------------------------------------------------")
    
    print("Generating response (running model forward passes)...")
    
    input_len = input_ids.shape[1]
    
    # Generate response
    with Timer("Generation forward loops"):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=40, # Shorten generation length to review every single token in detail
            temperature=0.3,   # Lower temperature for high determinism and easier explanation
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    generated_token_ids = output_ids[0][input_len:].tolist()
    
    print(f"\nGenerated {len(generated_token_ids)} tokens.")
    print("Below is the step-by-step breakdown of how each generated token builds the sentence:")
    print("=" * 80)
    print(f"{'INDEX':<6} | {'TOKEN ID':<8} | {'SUBWORD':<15} | {'SEMANTIC ROLE / DESCRIPTION'}")
    print("=" * 80)
    
    # Task 9: Explaining every generated token
    for idx, token_id in enumerate(generated_token_ids):
        # Decode the single token
        subword = tokenizer.decode([token_id])
        
        # Determine semantic/grammatical role of the subword
        cleaned_subword = subword.strip().lower()
        role = "Transition/Punctuation"
        
        if token_id == tokenizer.eos_token_id:
            role = "Special Token: EOS (End Of Sequence) - Tells generator loop to stop."
        elif cleaned_subword in ["the", "a", "an"]:
            role = "Definite/Indefinite Article - Specifies the noun following it."
        elif cleaned_subword in ["heart", "organ", "cardiac", "muscle"]:
            role = "Medical Subject Matter Noun - Refers to the central topic."
        elif cleaned_subword in ["is", "acts", "pumps", "serves", "function"]:
            role = "Action Verb / State Indicator - Describes the subject's functionality."
        elif cleaned_subword in ["to", "for", "in", "of", "with"]:
            role = "Preposition - Connects nouns and verbs to represent relationship."
        elif cleaned_subword in ["blood", "oxygen", "body", "circulatory", "system"]:
            role = "Medical Object Noun - Denotes target of the heart's action."
        elif cleaned_subword in [",", "."]:
            role = "Punctuation - Controls syntactic flow and pause boundaries."
        else:
            role = "Contextual content token contributing to the explanation."
            
        print(f"{idx:<6d} | {token_id:<8d} | {repr(subword):<15} | {role}")
        
    print("=" * 80)
    
    # Print complete generated response
    full_response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    print("\n[COMPLETE DECODED RESPONSE]:")
    print(full_response)
    print("\n---------------------------------------------------------------------")
    print("LEARNING REVIEW")
    print("---------------------------------------------------------------------")
    print("1. Prefill: The model processed the input prompt IDs simultaneously.")
    print("2. KV Cache: Key-Value parameters were cached to speed up the loop.")
    print("3. Autoregressive steps: For each generated token above, the model ran a")
    print("   forward pass, selected the token from logits, and fed it back as input.")
    print("=====================================================================")

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("\n!!! EXCEPTION CAUGHT IN MAIN !!!")
        print(traceback.format_exc())
        sys.exit(1)
