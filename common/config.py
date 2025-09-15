"""
Configuration loader for SpecECD project
"""
import toml
from pathlib import Path
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load configuration from config.toml"""
    config_path = Path(__file__).parent / "config.toml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return toml.load(f)
    else:
        # Default configuration if file doesn't exist
        return {
            "models": {
                "edge_model": "meta-llama/Llama-3.2-1B-Instruct",
                "cloud_model": "meta-llama/Llama-3.1-8B-Instruct",
                "max_edge_tokens": 5,
                "max_cloud_tokens": 10
            },
            "devices": {
                "edge_device": "cpu",
                "cloud_device": "auto"
            },
            "network": {
                "default_host": "localhost", 
                "default_port": 8765
            },
            "performance": {
                "temperature": 0.7,
                "repetition_penalty": 1.1
            }
        }

def get_edge_model_config() -> Dict[str, Any]:
    """Get edge model configuration"""
    config = load_config()
    return {
        "model_name": config["models"]["edge_model"],
        "device": config["devices"]["edge_device"],
        "max_tokens": config["models"]["max_edge_tokens"],
        "temperature": config["performance"]["temperature"],
        "repetition_penalty": config["performance"]["repetition_penalty"]
    }

def get_cloud_model_config() -> Dict[str, Any]:
    """Get cloud model configuration"""
    config = load_config()
    return {
        "model_name": config["models"]["cloud_model"], 
        "device": config["devices"]["cloud_device"],
        "max_tokens": config["models"]["max_cloud_tokens"],
        "temperature": config["performance"]["temperature"],
        "repetition_penalty": config["performance"]["repetition_penalty"]
    }

def get_network_config() -> Dict[str, Any]:
    """Get network configuration"""
    config = load_config()
    return {
        "host": config["network"]["default_host"],
        "port": config["network"]["default_port"]
    }
