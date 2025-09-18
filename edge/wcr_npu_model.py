"""
WCR ONNX Runtime GenAI NPU model wrapper for edge inference
Based on Windows Compatible Runtime (WCR) implementation
Provides NPU acceleration using ONNX Runtime GenAI with WinML execution providers
"""
import os
import sys
import json
import shutil
import logging
import tempfile
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from common.protocol import CreateTimestamp

try:
    import onnxruntime_genai as og
    ONNXRUNTIME_GENAI_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_GENAI_AVAILABLE = False
    og = None

# Configure logging
logging.basicConfig(level=logging.INFO)
g_logger = logging.getLogger(__name__)

class WCRNPUModel:
    """WCR ONNX Runtime GenAI NPU accelerated model for edge inference"""
    
    def __init__(self, model_path: str):
        """
        Initialize WCR NPU model
        
        Args:
            model_path: Path to the folder containing ONNX model files (WCR format)
        """
        self.m_model_path = Path(model_path)
        self.m_device = "WCR NPU (ONNX Runtime GenAI)"
        self.m_model = None
        self.m_tokenizer = None
        self.m_search_options = None
        self.m_execution_providers_registered = False
        
        if not ONNXRUNTIME_GENAI_AVAILABLE:
            raise ImportError(
                "ONNX Runtime GenAI not available. Install with:\n"
                "pip install onnxruntime-genai\n"
                "For WCR NPU support also install:\n"
                "pip install onnxruntime-winml winui3"
            )
        
        # Validate model path
        if not self.m_model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        if not self.m_model_path.is_dir():
            raise ValueError(f"Model path must be a directory: {model_path}")
        
        g_logger.info(f"Initializing WCR NPU model from: {self.m_model_path}")
        g_logger.info("Using ONNX Runtime GenAI with WinML execution providers")
        
    def _RegisterExecutionProviders(self) -> bool:
        """Register WinML execution providers for NPU support"""
        if self.m_execution_providers_registered:
            return True
            
        try:
            # Import WCR helper for execution provider registration
            from edge.wcr_winml_helper import register_execution_providers_to_onnxruntime_genai, is_qnn_provider_available
            
            g_logger.info("Registering WinML execution providers for NPU support...")
            register_execution_providers_to_onnxruntime_genai()
            self.m_execution_providers_registered = True
            
            # Check for NPU/QNN provider support using WCR helper
            if is_qnn_provider_available():
                g_logger.info("✓ NPU/QNN providers available after registration")
            else:
                g_logger.warning("No NPU/QNN providers detected after registration")
                
            return True
            
        except ImportError as e:
            g_logger.error(f"Failed to import WCR helper: {e}")
            g_logger.error("Make sure winui3 and onnxruntime-winml are installed")
            return False
        except Exception as e:
            g_logger.error(f"Failed to register execution providers: {e}")
            return False
    
    def _CheckModelFiles(self) -> bool:
        """Check if required ONNX model files are present (WCR format)"""
        required_files = [
            "config.json",
            "genai_config.json"
        ]
        
        # Check for at least one ONNX model file
        onnx_files = list(self.m_model_path.glob("*.onnx"))
        if not onnx_files:
            g_logger.error("No ONNX model files found in directory")
            return False
        
        # Check for required config files
        missing_files = []
        for file_name in required_files:
            file_path = self.m_model_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            g_logger.warning(f"Missing config files: {missing_files}")
            g_logger.info("Model may still work with default configurations")
        
        g_logger.info(f"Found {len(onnx_files)} ONNX model files:")
        for onnx_file in onnx_files:
            g_logger.info(f"  - {onnx_file.name}")
        
        return True
    
    def LoadModel(self) -> bool:
        """Load and initialize the WCR NPU model"""
        try:
            # Register execution providers first
            if not self._RegisterExecutionProviders():
                g_logger.error("Failed to register execution providers")
                return False
            
            # Check model files
            if not self._CheckModelFiles():
                g_logger.error("Model validation failed")
                return False
            
            g_logger.info("Loading WCR NPU model with ONNX Runtime GenAI...")
            
            # Load the model
            self.m_model = og.Model(str(self.m_model_path))
            g_logger.info("✓ ONNX model loaded successfully")
            
            # Load the tokenizer
            self.m_tokenizer = og.Tokenizer(self.m_model)
            g_logger.info("✓ Tokenizer loaded successfully")
            
            # Set up search options for generation
            self.m_search_options = og.GeneratorParams(self.m_model)
            
            # Configure generation parameters optimized for NPU
            self.m_search_options.set_search_options(
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                max_length=4096  # Match WCR configuration
            )
            
            g_logger.info("✓ WCR NPU model ready for inference")
            g_logger.info("Hardware: Qualcomm NPU/HTP acceleration via WinML")
            g_logger.info("Optimization: ONNX quantization with WCR runtime")
            
            return True
            
        except Exception as e:
            g_logger.error(f"Failed to load WCR NPU model: {e}")
            g_logger.error("Troubleshooting:")
            g_logger.error("1. Verify model path contains valid ONNX files")
            g_logger.error("2. Check config.json and genai_config.json are present")
            g_logger.error("3. Install: pip install onnxruntime-genai")
            g_logger.error("4. Install WCR support: pip install onnxruntime-winml winui3")
            g_logger.error("5. Ensure NPU drivers are installed and updated")
            return False
    
    def GenerateDraftTokensWithProbabilities(
        self, 
        prompt: str, 
        num_draft_tokens: int = 5,
        temperature: float = 0.8
    ) -> Tuple[List[str], List[float], float]:
        """
        Generate draft tokens with probabilities using WCR NPU acceleration
        
        Returns:
            (draft_tokens, draft_probabilities, inference_time)
        """
        if not self.m_model or not self.m_tokenizer:
            g_logger.error("WCR NPU model not loaded")
            return [], [], 0.0
        
        start_time = CreateTimestamp()
        
        try:
            g_logger.debug(f"WCR NPU generating {num_draft_tokens} tokens for: '{prompt[:50]}...'")
            
            # Encode the prompt
            input_tokens = self.m_tokenizer.encode(prompt)
            
            # Set up generation parameters
            params = og.GeneratorParams(self.m_model)
            params.set_search_options(
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                max_length=len(input_tokens) + num_draft_tokens
            )
            params.input_ids = input_tokens
            
            # Generate tokens using WCR NPU acceleration
            generator = og.Generator(self.m_model, params)
            
            draft_tokens = []
            draft_probs = []
            
            # Generate tokens one by one to track probabilities
            for i in range(num_draft_tokens):
                if generator.is_done():
                    break
                
                # Compute next token with NPU acceleration
                generator.compute_logits()
                
                # Get logits (if available) for probability calculation
                try:
                    logits = generator.get_logits()
                    if logits is not None and len(logits) > 0:
                        # Convert to numpy and compute probabilities
                        logits_array = np.array(logits, dtype=np.float32).flatten()
                        
                        # Apply temperature
                        if temperature != 1.0:
                            logits_array = logits_array / temperature
                        
                        # Compute softmax probabilities
                        exp_logits = np.exp(logits_array - np.max(logits_array))
                        probs = exp_logits / np.sum(exp_logits)
                        
                        # Get the selected token and its probability
                        generator.generate_next_token()
                        new_token_id = generator.get_next_tokens()[0]
                        token_prob = float(probs[new_token_id]) if new_token_id < len(probs) else 0.7
                    else:
                        # Fallback: generate token without detailed probability
                        generator.generate_next_token()
                        new_token_id = generator.get_next_tokens()[0]
                        token_prob = 0.7  # Placeholder probability
                        
                except Exception as prob_error:
                    g_logger.debug(f"Probability calculation failed: {prob_error}")
                    # Fallback: generate token without detailed probability
                    generator.generate_next_token()
                    new_token_id = generator.get_next_tokens()[0]
                    token_prob = 0.7  # Placeholder probability
                
                # Decode token
                token_text = self.m_tokenizer.decode([new_token_id])
                draft_tokens.append(token_text)
                draft_probs.append(token_prob)
                
                g_logger.debug(f"WCR NPU token {i+1}: '{token_text}' (p={token_prob:.3f})")
            
            inference_time = CreateTimestamp() - start_time
            
            g_logger.info(f"WCR NPU generated {len(draft_tokens)} draft tokens in {inference_time:.3f}s")
            g_logger.debug(f"Draft sequence: {''.join(draft_tokens)}")
            g_logger.debug(f"Average probability: {sum(draft_probs)/len(draft_probs):.3f}")
            
            return draft_tokens, draft_probs, inference_time
            
        except Exception as e:
            g_logger.error(f"WCR NPU draft generation failed: {e}")
            return [], [], CreateTimestamp() - start_time
    
    def GenerateText(self, prompt: str, max_tokens: int = 50) -> str:
        """
        Generate text using WCR NPU acceleration
        """
        if not self.m_model or not self.m_tokenizer:
            g_logger.error("WCR NPU model not loaded")
            return ""
        
        try:
            # Encode the prompt
            input_tokens = self.m_tokenizer.encode(prompt)
            
            # Set up generation parameters
            params = og.GeneratorParams(self.m_model)
            params.set_search_options(
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                max_length=len(input_tokens) + max_tokens
            )
            params.input_ids = input_tokens
            
            # Generate text with NPU acceleration
            output_tokens = og.generate(self.m_model, params)
            
            # Decode the full output
            output_text = self.m_tokenizer.decode(output_tokens[0])
            
            # Extract only the generated part (remove prompt)
            generated_text = output_text[len(prompt):]
            
            return generated_text
                
        except Exception as e:
            g_logger.error(f"WCR NPU text generation failed: {e}")
            return ""
    
    def GetModelInfo(self) -> dict:
        """Get information about the WCR NPU model"""
        info = {
            "model_path": str(self.m_model_path),
            "device": self.m_device,
            "backend": "WCR ONNX Runtime GenAI",
            "optimization": "NPU acceleration via WinML",
            "supported_providers": []
        }
        
        # Add provider information
        try:
            if ONNXRUNTIME_GENAI_AVAILABLE:
                # Use WCR helper to check provider support instead of direct API call
                from edge.wcr_winml_helper import get_available_providers_info
                provider_info = get_available_providers_info()
                info["supported_providers"] = provider_info.get("available_providers", [])
                info["provider_paths"] = provider_info.get("provider_paths", {})
                info["onnxruntime_genai_available"] = True
                info["execution_providers_registered"] = self.m_execution_providers_registered
            else:
                info["onnxruntime_genai_available"] = False
        except Exception as e:
            g_logger.debug(f"Could not get provider info: {e}")
            info["onnxruntime_genai_available"] = ONNXRUNTIME_GENAI_AVAILABLE
        
        return info

def IsWCRNPUAvailable() -> bool:
    """Check if WCR NPU support is available"""
    if not ONNXRUNTIME_GENAI_AVAILABLE:
        return False
    
    try:
        # Try to import WCR dependencies
        import winui3
        import onnxruntime_genai as og
        
        # Check if execution providers can be discovered
        from edge.wcr_winml_helper import _get_execution_provider_paths
        providers = _get_execution_provider_paths()
        
        # Look for NPU/QNN providers
        npu_providers = [name for name in providers.keys() if 'NPU' in name or 'QNN' in name]
        return len(npu_providers) > 0
        
    except ImportError:
        return False
    except Exception as e:
        g_logger.debug(f"WCR NPU availability check failed: {e}")
        return False

def ValidateWCRModelPath(model_path: str) -> bool:
    """Validate that the model path contains required WCR ONNX files"""
    try:
        path = Path(model_path)
        if not path.exists() or not path.is_dir():
            return False
        
        # Check for ONNX files
        onnx_files = list(path.glob("*.onnx"))
        if not onnx_files:
            return False
        
        # Check for basic config file
        config_file = path / "config.json"
        if not config_file.exists():
            g_logger.warning(f"config.json not found in {model_path}")
            # May still be valid, continue
        
        return True
    except Exception:
        return False