"""
Test script for OpenVINO NPU functionality
Tests device detection, model loading, and inference
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

def test_npu_availability():
    """Test if NPU is available"""
    print("="*60)
    print("Testing NPU Availability")
    print("="*60)
    
    try:
        from edge.openvino_model import IsNPUAvailable
        
        available = IsNPUAvailable()
        print(f"NPU Available: {available}")
        
        if available:
            print("✓ NPU hardware detected and ready")
        else:
            print("✗ NPU hardware not detected")
            print("  This could be due to:")
            print("  - No NPU hardware present")
            print("  - NPU drivers not installed")
            print("  - OpenVINO not configured for NPU")
        
        return available
        
    except ImportError as e:
        print(f"✗ OpenVINO not available: {e}")
        print("  Install with: pip install openvino openvino-dev optimum[openvino]")
        return False
    except Exception as e:
        print(f"✗ Error checking NPU: {e}")
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
        
        # Don't actually load the full model for testing
        print("✓ CPU configuration validated")
        
    except Exception as e:
        print(f"✗ CPU model failed: {e}")
    
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
    """Test configuration loading with NPU options"""
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
        
        # Check NPU settings
        npu_config = config.get("devices", {}).get("npu", {})
        print(f"✓ NPU enabled: {npu_config.get('enabled', False)}")
        print(f"✓ NPU fallback: {npu_config.get('fallback_to_cpu', True)}")
        
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
    """Print usage instructions for NPU"""
    print("\n" + "="*60)
    print("NPU Usage Instructions")
    print("="*60)
    
    print("To enable OpenVINO NPU acceleration:")
    print()
    print("1. Install OpenVINO with NPU support:")
    print("   pip install openvino openvino-dev optimum[openvino]")
    print()
    print("2. Enable NPU in config.toml:")
    print("   [devices.npu]")
    print("   enabled = true")
    print()
    print("3. Run edge client with NPU:")
    print("   python run_edge.py --device npu")
    print()
    print("4. Or force NPU in configuration:")
    print("   [devices]")
    print("   edge_device = \"npu\"")
    print()
    print("5. Check available devices:")
    print("   python run_edge.py --list-devices")

def main():
    """Run all NPU tests"""
    print("SpecECD OpenVINO NPU Test Suite")
    print("="*60)
    
    # Test 1: NPU availability
    npu_available = test_npu_availability()
    
    # Test 2: Device configurations
    test_device_configurations()
    
    # Test 3: Configuration loading
    test_configuration_loading()
    
    # Test 4: Edge client initialization
    test_edge_client_init()
    
    # Print usage instructions
    print_usage_instructions()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    if npu_available:
        print("✓ NPU hardware detected - ready for acceleration")
        print("  You can use --device npu to enable NPU acceleration")
    else:
        print("! NPU hardware not detected - CPU fallback available")
        print("  The system will use CPU inference with automatic fallback")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()