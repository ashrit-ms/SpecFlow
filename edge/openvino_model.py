"""
OpenVINO NPU model wrapper for edge inference
Provides NPU acceleration for draft token generation
"""
import numpy as np
import logging
from typing import List, Tuple, Optional
from pathlib import Path
import tempfile
import shutil

try:
    import openvino as ov
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    ov = None
    OVModelForCausalLM = None

from common.protocol import CreateTimestamp

# Configure logging
logging.basicConfig(level=logging.INFO)
g_logger = logging.getLogger(__name__)

class OpenVINONPUModel:
    """OpenVINO NPU accelerated model for edge inference"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
        """Initialize OpenVINO NPU model"""
        self.m_model_name = model_name
        self.m_device = "NPU"
        self.m_tokenizer = None
        self.m_model = None
        self.m_ov_model_path = None
        self.m_temp_dir = None
        
        if not OPENVINO_AVAILABLE:
            raise ImportError(
                "OpenVINO not available. Install with: pip install openvino openvino-dev optimum[openvino]"
            )
        
        g_logger.info(f"Initializing OpenVINO NPU model: {model_name}")
        
    def _CheckNPUAvailability(self) -> bool:
        """Check if NPU device is available"""
        try:
            core = ov.Core()
            available_devices = core.available_devices
            g_logger.info(f"Available OpenVINO devices: {available_devices}")
            
            # Check for NPU devices
            npu_devices = [device for device in available_devices if 'NPU' in device]
            if not npu_devices:
                g_logger.warning("No NPU devices found. Available devices: " + ", ".join(available_devices))
                return False
            
            g_logger.info(f"Found NPU devices: {npu_devices}")
            return True
            
        except Exception as e:
            g_logger.error(f"Error checking NPU availability: {e}")
            return False
    
    def LoadModel(self) -> bool:
        """Load and compile model for NPU"""
        try:
            # Check NPU availability first
            if not self._CheckNPUAvailability():
                g_logger.error("NPU not available, cannot load OpenVINO NPU model")
                return False
            
            # Load tokenizer
            g_logger.info("Loading tokenizer...")
            self.m_tokenizer = AutoTokenizer.from_pretrained(
                self.m_model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.m_tokenizer.pad_token is None:
                self.m_tokenizer.pad_token = self.m_tokenizer.eos_token
            
            # Create temporary directory for OpenVINO IR files
            self.m_temp_dir = tempfile.mkdtemp(prefix="openvino_model_")
            self.m_ov_model_path = Path(self.m_temp_dir) / "model"
            
            g_logger.info("Converting and loading model for NPU...")
            
            # Load model using Optimum with NPU device
            # This will automatically convert the model to OpenVINO IR format
            self.m_model = OVModelForCausalLM.from_pretrained(
                self.m_model_name,
                export=True,  # Convert to OpenVINO IR
                device=self.m_device,  # Target NPU
                trust_remote_code=True,
                cache_dir=self.m_temp_dir
            )
            
            g_logger.info("OpenVINO NPU model loaded successfully")
            return True
            
        except Exception as e:
            g_logger.error(f"Failed to load OpenVINO NPU model: {e}")
            self._Cleanup()
            return False
    
    def GenerateDraftTokensWithProbabilities(
        self, 
        prompt: str, 
        num_draft_tokens: int = 5,
        temperature: float = 0.8
    ) -> Tuple[List[str], List[float], float]:
        """
        Generate draft tokens with probabilities using NPU acceleration
        
        Returns:
            (draft_tokens, draft_probabilities, inference_time)
        """
        if not self.m_model or not self.m_tokenizer:
            g_logger.error("Model not loaded")
            return [], [], 0.0
        
        start_time = CreateTimestamp()
        
        try:
            # Encode prompt
            input_ids = self.m_tokenizer.encode(prompt, return_tensors="np")
            
            draft_tokens = []
            draft_probs = []
            
            # Generate tokens one by one to get probabilities
            current_input = input_ids
            
            for i in range(num_draft_tokens):
                # Get model outputs
                outputs = self.m_model(current_input)
                logits = outputs.logits[0, -1, :]  # Last position logits
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Convert to probabilities using softmax
                exp_logits = np.exp(logits - np.max(logits))  # Stable softmax
                probs = exp_logits / np.sum(exp_logits)
                
                # Sample token
                token_id = np.random.choice(len(probs), p=probs)
                token_prob = probs[token_id]
                
                # Decode token
                token = self.m_tokenizer.decode([token_id])
                
                # Store token and probability
                draft_tokens.append(token)
                draft_probs.append(float(token_prob))
                
                # Update input for next iteration
                current_input = np.concatenate([current_input, [[token_id]]], axis=1)
                
                g_logger.debug(f"NPU generated token {i+1}: '{token}' (p={token_prob:.3f})")
            
            inference_time = CreateTimestamp() - start_time
            
            g_logger.info(f"NPU generated {len(draft_tokens)} draft tokens with probabilities in {inference_time:.3f}s")
            g_logger.debug(f"Draft sequence: {''.join(draft_tokens)}")
            g_logger.debug(f"Average probability: {sum(draft_probs)/len(draft_probs):.3f}")
            
            return draft_tokens, draft_probs, inference_time
            
        except Exception as e:
            g_logger.error(f"NPU draft generation with probabilities failed: {e}")
            return [], [], CreateTimestamp() - start_time
    
    def GetModelInfo(self) -> dict:
        """Get information about the NPU model"""
        info = {
            "model_name": self.m_model_name,
            "device": self.m_device,
            "backend": "OpenVINO NPU",
            "model_path": str(self.m_ov_model_path) if self.m_ov_model_path else None
        }
        
        # Add NPU-specific information
        try:
            if OPENVINO_AVAILABLE:
                core = ov.Core()
                npu_devices = [device for device in core.available_devices if 'NPU' in device]
                info["available_npu_devices"] = npu_devices
        except Exception as e:
            g_logger.warning(f"Could not get NPU device info: {e}")
        
        return info
    
    def _Cleanup(self):
        """Clean up temporary files"""
        if self.m_temp_dir and Path(self.m_temp_dir).exists():
            try:
                shutil.rmtree(self.m_temp_dir)
                g_logger.info("Cleaned up temporary OpenVINO files")
            except Exception as e:
                g_logger.warning(f"Failed to cleanup temporary files: {e}")
    
    def __del__(self):
        """Destructor to cleanup resources"""
        self._Cleanup()

def IsNPUAvailable() -> bool:
    """Check if NPU is available for OpenVINO"""
    if not OPENVINO_AVAILABLE:
        return False
    
    try:
        core = ov.Core()
        available_devices = core.available_devices
        npu_devices = [device for device in available_devices if 'NPU' in device]
        return len(npu_devices) > 0
    except Exception:
        return False