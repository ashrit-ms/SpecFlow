"""
Test script for GPU functionality on edge inference
Tests CUDA availability, GPU memory, and model loading
"""
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
g_logger = logging.getLogger(__name__)

def test_gpu_availability():
    """Test if GPU/CUDA is available"""
    print("="*60)
    print("Testing GPU/CUDA Availability")
    print("="*60)
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            print(f"Number of GPUs: {gpu_count}")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")
                print(f"  Multiprocessors: {props.multi_processor_count}")
            
            print("✓ GPU hardware detected and ready")
            return True
        else:
            print("✗ CUDA not available")
            print("  This could be due to:")
            print("  - No NVIDIA GPU present")
            print("  - CUDA drivers not installed")
            print("  - PyTorch built without CUDA support")
            return False
        
    except ImportError as e:
        print(f"✗ PyTorch not available: {e}")
        print("  Install with: pip install torch torchvision torchaudio")
        return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False

def test_gpu_memory():
    """Test GPU memory allocation"""
    print("\n" + "="*60)
    print("Testing GPU Memory")
    print("="*60)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("✗ GPU not available for memory test")
            return False
        
        device = torch.cuda.current_device()
        print(f"Current GPU device: {device}")
        
        # Get memory info
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)
        
        print(f"Total GPU memory: {total_memory / 1e9:.2f} GB")
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Cached memory: {cached_memory / 1e9:.2f} GB")
        print(f"Free memory: {(total_memory - allocated_memory) / 1e9:.2f} GB")
        
        # Test simple allocation
        test_tensor = torch.randn(1000, 1000).cuda()
        new_allocated = torch.cuda.memory_allocated(device)
        print(f"After test allocation: {new_allocated / 1e9:.2f} GB")
        
        # Clean up
        del test_tensor
        torch.cuda.empty_cache()
        
        print("✓ GPU memory test passed")
        return True
        
    except Exception as e:
        print(f"✗ GPU memory test failed: {e}")
        return False

def test_device_configurations():
    """Test different device configurations"""
    print("\n" + "="*60)
    print("Testing Device Configurations")
    print("="*60)
    
    from edge.draft_model import EdgeDraftModel
    
    # Test CPU device
    print("\n--- Testing CPU Device ---")
    try:
        cpu_model = EdgeDraftModel(device="cpu")
        print("✓ CPU model instance created")
        
    except Exception as e:
        print(f"✗ CPU model failed: {e}")
    
    # Test GPU device
    print("\n--- Testing GPU Device ---")
    try:
        gpu_model = EdgeDraftModel(device="gpu")
        print("✓ GPU model instance created")
        
        # Check if it falls back to CPU
        if gpu_model.m_device == "cpu":
            print("! GPU requested but fell back to CPU")
        else:
            print("✓ GPU configuration validated")
            print(f"✓ CUDA device: {gpu_model.m_cuda_device}")
            
    except Exception as e:
        print(f"✗ GPU model failed: {e}")
    
    # Test NPU device
    print("\n--- Testing NPU Device ---")
    try:
        npu_model = EdgeDraftModel(device="npu")
        print("✓ NPU model instance created")
        
        # Check if it falls back to CPU
        if npu_model.m_device == "cpu":
            print("! NPU requested but fell back to CPU")
        else:
            print("✓ NPU configuration validated")
            
    except Exception as e:
        print(f"✗ NPU model failed: {e}")

def test_configuration_loading():
    """Test configuration loading with GPU options"""
    print("\n" + "="*60)
    print("Testing Configuration Loading")
    print("="*60)
    
    try:
        from common.config import get_edge_model_config, load_config
        
        # Test config loading
        config = load_config()
        print("✓ Configuration loaded")
        
        # Test edge model config
        edge_config = get_edge_model_config()
        print(f"✓ Edge device config: {edge_config['device']}")
        print(f"✓ Model name: {edge_config['model_name']}")
        
        # Check GPU settings
        gpu_config = config.get("devices", {}).get("gpu", {})
        print(f"✓ GPU enabled: {gpu_config.get('enabled', False)}")
        print(f"✓ GPU device ID: {gpu_config.get('device_id', 0)}")
        
        # Check NPU settings
        npu_config = config.get("devices", {}).get("npu", {})
        print(f"✓ NPU enabled: {npu_config.get('enabled', False)}")
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")

def test_edge_client_init():
    """Test edge client initialization with different devices"""
    print("\n" + "="*60)
    print("Testing Edge Client Initialization")
    print("="*60)
    
    try:
        from edge.client import EdgeClient
        
        # Test CPU client
        print("\n--- Testing CPU Client ---")
        cpu_client = EdgeClient(device="cpu")
        print("✓ CPU edge client created")
        
        # Test GPU client
        print("\n--- Testing GPU Client ---")
        gpu_client = EdgeClient(device="gpu")
        print(f"✓ GPU edge client created (actual device: {gpu_client.m_device})")
        
        # Test NPU client
        print("\n--- Testing NPU Client ---")
        npu_client = EdgeClient(device="npu")
        print(f"✓ NPU edge client created (actual device: {npu_client.m_device})")
        
        # Test default client (from config)
        print("\n--- Testing Default Client ---")
        default_client = EdgeClient()
        print(f"✓ Default edge client created (device: {default_client.m_device})")
        
    except Exception as e:
        print(f"✗ Edge client test failed: {e}")

def print_usage_instructions():
    """Print usage instructions for GPU"""
    print("\n" + "="*60)
    print("GPU Usage Instructions")
    print("="*60)
    
    print("To enable GPU acceleration on edge:")
    print()
    print("1. Ensure CUDA is installed:")
    print("   - Install NVIDIA drivers")
    print("   - Install CUDA toolkit")
    print("   - Install PyTorch with CUDA support:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print()
    print("2. Enable GPU in config.toml:")
    print("   [devices.gpu]")
    print("   enabled = true")
    print()
    print("3. Run edge client with GPU:")
    print("   python run_edge.py --device gpu")
    print()
    print("4. Or force GPU in configuration:")
    print("   [devices]")
    print("   edge_device = \"gpu\"")
    print()
    print("5. Check available devices:")
    print("   python run_edge.py --list-devices")

def main():
    """Run all GPU tests"""
    print("SpecECD GPU Test Suite")
    print("="*60)
    
    # Test 1: GPU availability
    gpu_available = test_gpu_availability()
    
    # Test 2: GPU memory (only if GPU available)
    if gpu_available:
        test_gpu_memory()
    
    # Test 3: Device configurations
    test_device_configurations()
    
    # Test 4: Configuration loading
    test_configuration_loading()
    
    # Test 5: Edge client initialization
    test_edge_client_init()
    
    # Print usage instructions
    print_usage_instructions()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    if gpu_available:
        print("✓ GPU hardware detected - ready for acceleration")
        print("  You can use --device gpu to enable GPU acceleration")
        print("  This will provide faster inference than CPU")
    else:
        print("! GPU hardware not detected - CPU/NPU fallback available")
        print("  The system will use CPU or NPU inference")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()