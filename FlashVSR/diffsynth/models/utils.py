import torch
from os.path import isfile
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import hf_hub_download


def get_model_path(model_id, cache_dir=None, file_name=None):
    if isfile(model_id):
        return model_id
    if file_name is None:
        file_name = "diffusion_pytorch_model.bin"
    return hf_hub_download(repo_id=model_id, filename=file_name, cache_dir=cache_dir)


def is_safetensors(path):
    return path.endswith(".safetensors")


def get_state_dict_from_checkpoint(checkpoint):
    state_dict = checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    return state_dict


def get_state_dict(model_id_or_path, cache_dir=None, file_name=None, torch_dtype=None):
    if isfile(model_id_or_path):
        if is_safetensors(model_id_or_path):
            state_dict = load_state_dict_from_safetensors(model_id_or_path, torch_dtype=torch_dtype)
        else:
            state_dict = load_state_dict_from_bin(model_id_or_path, torch_dtype=torch_dtype)
    else:
        model_path = get_model_path(model_id_or_path, cache_dir, file_name)
        if is_safetensors(model_path):
            state_dict = load_state_dict_from_safetensors(model_path, torch_dtype=torch_dtype)
        else:
            state_dict = load_state_dict_from_bin(model_path, torch_dtype=torch_dtype)
    return state_dict


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = load_safetensors(file_path)
    if torch_dtype is not None:
        for k, v in state_dict.items():
            if v.dtype == torch.float32:
                state_dict[k] = v.to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    """
    Load a state dict from a .bin or .safetensors file.
    This is the modified function.
    """
    # This logic correctly handles both file types.
    if file_path.endswith(".safetensors"):
        state_dict = load_safetensors(file_path, device="cpu")
    else:
        # Use the original torch.load for other file types (.bin, .pt, etc.)
        state_dict = torch.load(file_path, map_location="cpu", weights_only=False)

    if torch_dtype is not None:
        for k, v in state_dict.items():
            if v.dtype == torch.float32:
                state_dict[k] = v.to(torch_dtype)
    return state_dict


def set_module_from_state_dict(module, state_dict, prefix=""):
    if not prefix.endswith(".") and len(prefix) > 0:
        prefix += "."
    
    # Load module parameters
    module_keys = [k[len(prefix):] for k in state_dict.keys() if k.startswith(prefix)]
    missing_keys = []
    for k, v in module.named_parameters():
        if k not in module_keys:
            missing_keys.append(k)
    if len(missing_keys) > 0:
        print(f"Warning: The following parameters are not loaded: {missing_keys}")
    
    # Update module
    module.load_state_dict({k: state_dict[prefix+k] for k in module_keys}, strict=False)


def set_module_from_state_dict_by_name(module, state_dict, name):
    prefix = name
    if name is not None and len(name) > 0:
        prefix += "."
    
    # Load module parameters
    state_dict_keys = [k for k in state_dict.keys() if (name is None) or k.startswith(prefix)]
    module_keys = {k for k, v in module.named_parameters()}
    
    # Check missing keys
    missing_keys = [k for k in module_keys if (prefix + k) not in state_dict_keys]
    if len(missing_keys) > 0:
        print(f"Warning: The following parameters are not loaded: {missing_keys}")
    
    # Update module
    sub_state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k in state_dict_keys}
    module.load_state_dict(sub_state_dict, strict=False)
