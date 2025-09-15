"""
Edge component - Draft model implementation for speculative decoding
Runs a small language model (<3B parameters) to generate draft tokens
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List, Optional, Tuple
import time
import logging
from common.protocol import PerformanceMetrics, CreateTimestamp

# Configure logging
logging.basicConfig(level=logging.INFO)
g_logger = logging.getLogger(__name__)

class EdgeDraftModel:
    """Draft model running on edge device"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
        """
        Initialize draft model with a small, efficient model from same family as target
        Using Llama-3.2-1B-Instruct (1B parameters) - same family as cloud model
        This ensures compatible tokenization and better acceptance rates
        """
        self.m_model_name = model_name
        self.m_device = "cpu"  # Force CPU for edge device
        self.m_tokenizer = None
        self.m_model = None
        self.m_generation_config = None
        
        g_logger.info(f"Initializing edge draft model: {model_name}")
        g_logger.info(f"Using device: {self.m_device} (forced CPU for edge)")
        
    def LoadModel(self) -> bool:
        """Load the draft model and tokenizer"""
        try:
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
            ).to(self.m_device)
            
            # Configure generation parameters
            self.m_generation_config = GenerationConfig(
                max_new_tokens=10,  # Generate few tokens as draft
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.m_tokenizer.pad_token_id,
                eos_token_id=self.m_tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            g_logger.info("Edge draft model loaded successfully")
            return True
            
        except Exception as e:
            g_logger.error(f"Failed to load edge model: {e}")
            return False
    
    def GenerateDraftTokensWithProbabilities(
        self, 
        prompt: str, 
        num_draft_tokens: int = 5,
        temperature: float = 0.8
    ) -> Tuple[List[str], List[float], float]:
        """
        CORRECTED: Generate draft tokens WITH their probabilities
        This is essential for proper speculative decoding verification
        
        Returns:
            (draft_tokens, draft_probabilities, inference_time)
        """
        if not self.m_model or not self.m_tokenizer:
            g_logger.error("Model not loaded")
            return [], [], 0.0
        
        start_time = CreateTimestamp()
        
        try:
            # Encode initial prompt
            input_ids = self.m_tokenizer.encode(prompt, return_tensors="pt").to(self.m_device)
            
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
                    current_input = torch.cat([current_input, torch.tensor([[token_id]]).to(self.m_device)], dim=1)
                    
                    g_logger.debug(f"Generated token {i+1}: '{token}' (p={token_prob:.3f})")
            
            inference_time = CreateTimestamp() - start_time
            
            g_logger.info(f"Generated {len(draft_tokens)} draft tokens with probabilities in {inference_time:.3f}s")
            g_logger.debug(f"Draft sequence: {''.join(draft_tokens)}")
            g_logger.debug(f"Average probability: {sum(draft_probs)/len(draft_probs):.3f}")
            
            return draft_tokens, draft_probs, inference_time
            
        except Exception as e:
            g_logger.error(f"Draft generation with probabilities failed: {e}")
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
        if not self.m_model:
            return {}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.m_model.parameters())
        trainable_params = sum(p.numel() for p in self.m_model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.m_model_name,
            "device": self.m_device,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "memory_footprint": f"{total_params * 2 / 1e9:.2f}GB"  # Approximate for fp16
        }
