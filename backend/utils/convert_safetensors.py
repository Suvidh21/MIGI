import os
import sys
import json
import struct
import torch
from tqdm import tqdm

def convert_safetensors_to_bin(snapshot_dir: str):
    """
    Parses Qwen's model.safetensors in pure Python/PyTorch buffer format
    and saves it as a standard PyTorch pickled file (pytorch_model.bin).
    
    Why it is needed:
        Some Windows environments crash during safetensors memory-mapping due to OS/filesystem
        access restrictions. Pickled state dicts load via standard file streams without memory mapping.
    """
    safetensors_path = os.path.join(snapshot_dir, "model.safetensors")
    bin_path = os.path.join(snapshot_dir, "pytorch_model.bin")
    
    if not os.path.exists(safetensors_path):
        print(f"Error: model.safetensors not found at {safetensors_path}")
        return False
        
    print(f"Reading model.safetensors: {safetensors_path}")
    
    with open(safetensors_path, "rb") as f:
        # Read header size
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            print("Error: Invalid safetensors file (too small).")
            return False
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        
        # Read JSON header
        header_bytes = f.read(header_size)
        header_json = json.loads(header_bytes.decode("utf-8"))
        
        state_dict = {}
        data_start_position = 8 + header_size
        
        # Mapping safetensors string dtypes to PyTorch dtypes
        dtype_map = {
            "BF16": torch.bfloat16,
            "F16": torch.float16,
            "F32": torch.float32,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8
        }
        
        print("Converting weight tensors...")
        # Loop through keys (skipping the special '__metadata__' key if present)
        keys = [k for k in header_json.keys() if k != "__metadata__"]
        
        for key in tqdm(keys):
            meta = header_json[key]
            dtype_str = meta["dtype"]
            shape = meta["shape"]
            start_offset, end_offset = meta["data_offsets"]
            
            # Resolve PyTorch dtype
            pt_dtype = dtype_map.get(dtype_str)
            if pt_dtype is None:
                raise ValueError(f"Unsupported dtype: {dtype_str} for key {key}")
                
            # Seek and read raw bytes
            tensor_start_pos = data_start_position + start_offset
            tensor_len = end_offset - start_offset
            
            f.seek(tensor_start_pos)
            tensor_bytes = f.read(tensor_len)
            
            # Convert bytes directly to PyTorch CPU tensor and reshape
            tensor = torch.frombuffer(tensor_bytes, dtype=pt_dtype).clone()
            
            # Handle empty shapes (scalars)
            if shape:
                tensor = tensor.view(shape)
                
            state_dict[key] = tensor
            
    print(f"Saving standard PyTorch state dict to: {bin_path}...")
    torch.save(state_dict, bin_path)
    print("Weight conversion completed successfully!")
    return True

if __name__ == "__main__":
    # Path to Qwen 2.5 snapshot in cache
    SNAPSHOT_DIR = "D:/Suvidh/Suvidh/DR.MIGI/models/cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
    success = convert_safetensors_to_bin(SNAPSHOT_DIR)
    sys.exit(0 if success else 1)
