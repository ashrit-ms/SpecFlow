"""
Edge component - Draft model implementation for speculative decoding
Runs a small language model (<3B parameters) to generate draft tokens
Supports both CPU and OpenVINO NPU acceleration
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List, Optional, Tuple
import time
import logging
from common.protocol import PerformanceMetrics, CreateTimestamp

# Try to import OpenVINO NPU support
try:
    from edge.openvino_model import OpenVINONPUModel, IsNPUAvailable
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    OpenVINONPUModel = None
    IsNPUAvailable = lambda: False

# Configure logging
logging.basicConfig(level=logging.INFO)
g_logger = logging.getLogger(__name__)

class EdgeDraftModel:
    """Draft model running on edge device with CPU, GPU, or NPU support"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct", device: str = "cpu"):
        """
        Initialize draft model with device selection
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ("cpu", "gpu", or "npu")
        """
        self.m_model_name = model_name
        self.m_device = device.lower()
        self.m_tokenizer = None
        self.m_model = None
        self.m_generation_config = None
        self.m_npu_model = None
        self.m_cuda_device = None
        
        g_logger.info(f"Initializing edge draft model: {model_name}")
        g_logger.info(f"Target device: {self.m_device}")
        
        # Validate device selection
        if self.m_device == "gpu":
            if not torch.cuda.is_available():
                g_logger.error("GPU device requested but CUDA not available")
                g_logger.info("Falling back to CPU device")
                self.m_device = "cpu"
            else:
                # Set CUDA device
                self.m_cuda_device = f"cuda:0"  # Default to first GPU
                g_logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
                g_logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        elif self.m_device == "npu":
            if not OPENVINO_AVAILABLE:
                g_logger.error("NPU device requested but OpenVINO not available")
                g_logger.info("Install OpenVINO with: pip install openvino openvino-dev optimum[openvino]")
                raise RuntimeError("OpenVINO NPU support not available")
            
            if not IsNPUAvailable():
                g_logger.error("NPU device requested but no NPU hardware detected")
                g_logger.info("Falling back to CPU device")
                self.m_device = "cpu"
        
        if self.m_device not in ["cpu", "gpu", "npu"]:
            g_logger.warning(f"Unknown device '{device}', falling back to CPU")
            self.m_device = "cpu"
        
    def LoadModel(self) -> bool:
        """Load the draft model and tokenizer based on device selection"""
        try:
            if self.m_device == "npu":
                return self._LoadNPUModel()
            elif self.m_device == "gpu":
                return self._LoadGPUModel()
            else:
                return self._LoadCPUModel()
                
        except Exception as e:
            g_logger.error(f"Failed to load edge model: {e}")
            return False
    
    def _LoadNPUModel(self) -> bool:
        """Load model for NPU inference using OpenVINO"""
        g_logger.info("Loading model for NPU inference...")
        
        # Create and load NPU model (OpenVINO wrapper handles model selection)
        self.m_npu_model = OpenVINONPUModel(self.m_model_name)
        
        if not self.m_npu_model.LoadModel():
            g_logger.error("Failed to load NPU model")
            g_logger.info("NPU loading failed. Common causes:")
            g_logger.info("1. NPU driver/hardware compatibility (update Intel drivers)")
            g_logger.info("2. OpenVINO version compatibility")
            g_logger.info("3. Model format issues (now using pre-converted model)")
            g_logger.info("Recommended fallbacks:")
            g_logger.info("  python run_tests.py --device cpu")
            g_logger.info("  python run_tests.py --device gpu")
            return False
        
        # Get tokenizer from NPU model
        self.m_tokenizer = self.m_npu_model.m_tokenizer
        
        g_logger.info("NPU model loaded successfully")
        return True
    
    def _LoadGPUModel(self) -> bool:
        """Load model for GPU inference using PyTorch CUDA"""
        g_logger.info("Loading model for GPU inference...")
        
        # Load tokenizer
        self.m_tokenizer = AutoTokenizer.from_pretrained(
            self.m_model_name,
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.m_tokenizer.pad_token is None:
            self.m_tokenizer.pad_token = self.m_tokenizer.eos_token
        
        # Load model with GPU optimizations
        g_logger.info(f"Loading model to {self.m_cuda_device}...")
        self.m_model = AutoModelForCausalLM.from_pretrained(
            self.m_model_name,
            torch_dtype=torch.float16,  # Use float16 for GPU memory efficiency
            device_map="auto",  # Automatically distribute across available GPUs
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(self.m_cuda_device)
        
        # Configure generation parameters optimized for GPU
        self.m_generation_config = GenerationConfig(
            max_new_tokens=10,  # Generate few tokens as draft
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.m_tokenizer.pad_token_id,
            eos_token_id=self.m_tokenizer.eos_token_id,
            repetition_penalty=1.1,
            use_cache=True  # Enable KV cache for faster generation
        )
        
        # Log GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            cached = torch.cuda.memory_reserved(0) / 1e9
            g_logger.info(f"GPU memory allocated: {allocated:.2f} GB")
            g_logger.info(f"GPU memory cached: {cached:.2f} GB")
        
        g_logger.info("GPU model loaded successfully")
        return True
    
    def _LoadCPUModel(self) -> bool:
        """Load model for CPU inference using PyTorch"""
        g_logger.info("Loading model for CPU inference...")
        
        # Load tokenizer
        self.m_tokenizer = AutoTokenizer.from_pretrained(
            self.m_model_name,
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.m_tokenizer.pad_token is None:
            self.m_tokenizer.pad_token = self.m_tokenizer.eos_token
        
        # Load model with CPU optimizations
        self.m_model = AutoModelForCausalLM.from_pretrained(
            self.m_model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None,  # Load on CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to("cpu")
        
        # Configure generation parameters
        self.m_generation_config = GenerationConfig(
            max_new_tokens=10,  # Generate few tokens as draft
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.m_tokenizer.pad_token_id,
            eos_token_id=self.m_tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
        
        g_logger.info("CPU model loaded successfully")
        return True
    
    def GenerateDraftTokensWithProbabilities(
        self, 
        prompt: str, 
        num_draft_tokens: int = 5,
        temperature: float = 0.8
    ) -> Tuple[List[str], List[float], float]:
        """
        Generate draft tokens WITH their probabilities
        Routes to appropriate backend (CPU PyTorch, GPU PyTorch, or NPU OpenVINO)
        
        Returns:
            (draft_tokens, draft_probabilities, inference_time)
        """
        if self.m_device == "npu" and self.m_npu_model:
            return self.m_npu_model.GenerateDraftTokensWithProbabilities(
                prompt, num_draft_tokens, temperature
            )
        elif self.m_device == "gpu":
            return self._GenerateGPUTokensWithProbabilities(
                prompt, num_draft_tokens, temperature
            )
        else:
            return self._GenerateCPUTokensWithProbabilities(
                prompt, num_draft_tokens, temperature
            )
    
    def _GenerateCPUTokensWithProbabilities(
        self, 
        prompt: str, 
        num_draft_tokens: int = 5,
        temperature: float = 0.8
    ) -> Tuple[List[str], List[float], float]:
        """
        Generate draft tokens with probabilities using CPU PyTorch
        """
        if not self.m_model or not self.m_tokenizer:
            g_logger.error("CPU model not loaded")
            return [], [], 0.0
        
        start_time = CreateTimestamp()
        
        try:
            # Encode initial prompt
            input_ids = self.m_tokenizer.encode(prompt, return_tensors="pt").to("cpu")
            
            draft_tokens = []
            draft_probs = []
            
            with torch.no_grad():
                current_input = input_ids
                
                # Generate tokens one by one, tracking probabilities
                for i in range(num_draft_tokens):
                    # Get logits for next token
                    outputs = self.m_model(current_input)
                    logits = outputs.logits[0, -1, :]  # Last position logits
                    
                    # Apply temperature
                    if temperature != 1.0:
                        logits = logits / temperature
                    
                    # Convert to probabilities
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    
                    # Sample token
                    token_id = torch.multinomial(probs, 1).item()
                    token_prob = probs[token_id].item()
                    
                    # Decode token
                    token = self.m_tokenizer.decode([token_id])
                    
                    # Store token and its probability
                    draft_tokens.append(token)
                    draft_probs.append(token_prob)
                    
                    # Update input for next iteration (autoregressive)
                    current_input = torch.cat([current_input, torch.tensor([[token_id]]).to("cpu")], dim=1)
                    
                    g_logger.debug(f"Generated token {i+1}: '{token}' (p={token_prob:.3f})")
            
            inference_time = CreateTimestamp() - start_time
            
            g_logger.info(f"CPU generated {len(draft_tokens)} draft tokens with probabilities in {inference_time:.3f}s")
            g_logger.debug(f"Draft sequence: {''.join(draft_tokens)}")
            g_logger.debug(f"Average probability: {sum(draft_probs)/len(draft_probs):.3f}")
            
            return draft_tokens, draft_probs, inference_time
            
        except Exception as e:
            g_logger.error(f"CPU draft generation with probabilities failed: {e}")
            return [], [], CreateTimestamp() - start_time
    
    def _GenerateGPUTokensWithProbabilities(
        self, 
        prompt: str, 
        num_draft_tokens: int = 5,
        temperature: float = 0.8
    ) -> Tuple[List[str], List[float], float]:
        """
        Generate draft tokens with probabilities using GPU PyTorch
        """
        if not self.m_model or not self.m_tokenizer:
            g_logger.error("GPU model not loaded")
            return [], [], 0.0
        
        start_time = CreateTimestamp()
        
        try:
            # Encode initial prompt
            input_ids = self.m_tokenizer.encode(prompt, return_tensors="pt").to(self.m_cuda_device)
            
            draft_tokens = []
            draft_probs = []
            
            with torch.no_grad():
                current_input = input_ids
                
                # Generate tokens one by one, tracking probabilities
                for i in range(num_draft_tokens):
                    # Get logits for next token
                    outputs = self.m_model(current_input)
                    logits = outputs.logits[0, -1, :]  # Last position logits
                    
                    # Apply temperature
                    if temperature != 1.0:
                        logits = logits / temperature
                    
                    # Convert to probabilities
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    
                    # Sample token
                    token_id = torch.multinomial(probs, 1).item()
                    token_prob = probs[token_id].item()
                    
                    # Decode token
                    token = self.m_tokenizer.decode([token_id])
                    
                    # Store token and its probability
                    draft_tokens.append(token)
                    draft_probs.append(token_prob)
                    
                    # Update input for next iteration (autoregressive)
                    current_input = torch.cat([current_input, torch.tensor([[token_id]]).to(self.m_cuda_device)], dim=1)
                    
                    g_logger.debug(f"GPU generated token {i+1}: '{token}' (p={token_prob:.3f})")
            
            inference_time = CreateTimestamp() - start_time
            
            g_logger.info(f"GPU generated {len(draft_tokens)} draft tokens with probabilities in {inference_time:.3f}s")
            g_logger.debug(f"Draft sequence: {''.join(draft_tokens)}")
            g_logger.debug(f"Average probability: {sum(draft_probs)/len(draft_probs):.3f}")
            
            return draft_tokens, draft_probs, inference_time
            
        except Exception as e:
            g_logger.error(f"GPU draft generation with probabilities failed: {e}")
            return [], [], CreateTimestamp() - start_time
    
    def GenerateDraftTokens(self, prompt: str, num_draft_tokens: int = 5) -> Tuple[List[str], float]:
        """
        Legacy interface for backward compatibility
        Note: This doesn't return probabilities, so verification will be suboptimal
        """
        g_logger.warning("Using legacy interface - probabilities not tracked, verification will be suboptimal")
        
        draft_tokens, draft_probs, inference_time = self.GenerateDraftTokensWithProbabilities(
            prompt, num_draft_tokens
        )
        
        return draft_tokens, inference_time
        """
        Generate draft tokens for the given prompt
        Returns (draft_tokens, inference_time)
        """
        if not self.m_model or not self.m_tokenizer:
            g_logger.error("Model not loaded")
            return [], 0.0
        
        start_time = CreateTimestamp()
        
        try:
            # Tokenize input
            inputs = self.m_tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.m_device)
            
            # Generate text (not individual tokens)
            with torch.no_grad():
                outputs = self.m_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=min(num_draft_tokens * 3, 30),  # Generate more text
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.m_tokenizer.pad_token_id,
                    eos_token_id=self.m_tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    use_cache=False
                )
            
            # Extract new text only (exclude input)
            input_length = inputs.input_ids.shape[1]
            new_token_ids = outputs[0][input_length:]
            
            # Decode the generated text
            generated_text = self.m_tokenizer.decode(new_token_ids, skip_special_tokens=True)
            
            # Split text into meaningful chunks (words/tokens)
            draft_tokens = []
            if generated_text.strip():
                # Simple approach: split by spaces and return as individual tokens
                words = generated_text.strip().split()
                for word in words[:num_draft_tokens]:
                    # Add space before word (except first) to maintain formatting
                    if draft_tokens:
                        draft_tokens.append(" " + word)
                    else:
                        draft_tokens.append(word)
                
                # Alternative: return the entire generated text as one token for better coherence
                if not draft_tokens:
                    draft_tokens = [generated_text.strip()]
            
            inference_time = CreateTimestamp() - start_time
            
            g_logger.info(f"Generated {len(draft_tokens)} draft tokens in {inference_time:.3f}s")
            g_logger.debug(f"Draft text: '{generated_text.strip()}'")
            
            return draft_tokens, inference_time
            
        except Exception as e:
            g_logger.error(f"Draft generation failed: {e}")
            return [], CreateTimestamp() - start_time

    def GetModelInfo(self) -> dict:
        """Get information about the loaded model"""
        base_info = {
            "model_name": self.m_model_name,
            "device": self.m_device,
        }
        
        if self.m_device == "npu" and self.m_npu_model:
            # Get NPU-specific info
            npu_info = self.m_npu_model.GetModelInfo()
            base_info.update(npu_info)
        elif self.m_device == "gpu" and self.m_model:
            # Count parameters for GPU model
            total_params = sum(p.numel() for p in self.m_model.parameters())
            trainable_params = sum(p.numel() for p in self.m_model.parameters() if p.requires_grad)
            
            # Get GPU memory info
            gpu_info = {}
            if torch.cuda.is_available() and self.m_cuda_device:
                device_id = int(self.m_cuda_device.split(':')[1])
                gpu_info = {
                    "gpu_name": torch.cuda.get_device_name(device_id),
                    "gpu_memory_total": f"{torch.cuda.get_device_properties(device_id).total_memory / 1e9:.1f}GB",
                    "gpu_memory_allocated": f"{torch.cuda.memory_allocated(device_id) / 1e9:.2f}GB",
                    "gpu_memory_cached": f"{torch.cuda.memory_reserved(device_id) / 1e9:.2f}GB"
                }
            
            base_info.update({
                "backend": "PyTorch GPU",
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "memory_footprint": f"{total_params * 2 / 1e9:.2f}GB",  # Approximate for fp16
                **gpu_info
            })
        elif self.m_model:
            # Count parameters for CPU model
            total_params = sum(p.numel() for p in self.m_model.parameters())
            trainable_params = sum(p.numel() for p in self.m_model.parameters() if p.requires_grad)
            
            base_info.update({
                "backend": "PyTorch CPU",
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "memory_footprint": f"{total_params * 2 / 1e9:.2f}GB"  # Approximate for fp16
            })
        
        return base_info
