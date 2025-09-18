"""
Main entry point for running edge client
Run this on the edge machine (laptop)
Supports CPU, GPU (CUDA), and WCR NPU acceleration
"""
import asyncio
import sys
import argparse
import logging
from edge.client import EdgeClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
g_logger = logging.getLogger(__name__)

def ParseArguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SpecECD Edge Client')
    parser.add_argument('--host', default='localhost', 
                       help='Cloud server host (default: localhost)')
    parser.add_argument('--port', type=int, default=8765,
                       help='Cloud server port (default: 8765)')
    parser.add_argument('--device', choices=['cpu', 'gpu', 'npu'], default=None,
                       help='Edge device type: cpu, gpu, or npu (default: from config)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to ONNX model folder (required for --device npu)')
    parser.add_argument('--list-devices', action='store_true',
                       help='List available devices and exit')
    
    return parser.parse_args()

def ListAvailableDevices():
    """List available devices for edge inference"""
    print("Available edge devices:")
    print("  cpu - PyTorch CPU inference (always available)")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  gpu - PyTorch GPU inference (CUDA device {i}: {gpu_name}, {gpu_memory:.1f}GB)")
        else:
            print("  gpu - PyTorch GPU inference (CUDA not available)")
    except ImportError:
        print("  gpu - PyTorch GPU inference (PyTorch not installed)")
    
    # Check NPU availability
    try:
        from edge.wcr_npu_model import IsWCRNPUAvailable
        if IsWCRNPUAvailable():
            print("  npu - WCR NPU acceleration (available)")
        else:
            print("  npu - WCR NPU acceleration (not available)")
    except ImportError:
        print("  npu - WCR NPU acceleration (not installed)")
        print("        Install with: pip install -r requirements-edge-wcr-npu.txt")

async def RunEdgeClient():
    """Main function to run edge client"""
    args = ParseArguments()
    
    # List devices if requested
    if args.list_devices:
        ListAvailableDevices()
        return
    
    g_logger.info(f"Starting edge client, connecting to {args.host}:{args.port}")
    if args.device:
        g_logger.info(f"Using device: {args.device}")
        if args.model_path:
            g_logger.info(f"Model path: {args.model_path}")
    else:
        g_logger.info("Using device from configuration")
    
    # Validate NPU requirements
    if args.device == 'npu':
        if not args.model_path:
            g_logger.error("--model-path is required when using --device npu")
            g_logger.error("Example: python run_edge.py --device npu --model-path /path/to/onnx/model")
            return
        
        try:
            from edge.wcr_npu_model import ValidateWCRModelPath
            if not ValidateWCRModelPath(args.model_path):
                g_logger.error(f"Invalid WCR ONNX model path: {args.model_path}")
                g_logger.error("Path must contain ONNX files and config.json")
                return
        except ImportError:
            g_logger.error("WCR NPU support not available")
            g_logger.error("Install with: pip install -r requirements-edge-wcr-npu.txt")
            return
    
    # Create and initialize edge client
    edge_client = EdgeClient(args.host, args.port, args.device, args.model_path)
    
    if not await edge_client.Initialize():
        g_logger.error("Failed to initialize edge client")
        return
    
    # Display model information
    model_info = edge_client.GetModelInfo()
    g_logger.info(f"Edge model info: {model_info}")
    
    # Connect to cloud
    if not await edge_client.ConnectToCloud():
        g_logger.error("Failed to connect to cloud server")
        return
    
    try:
        # Interactive mode - keep generating text
        while True:
            prompt = input("\nEnter prompt (or 'quit' to exit): ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
            
            g_logger.info("Generating response...")
            generated_text, metrics = await edge_client.GenerateWithSpeculation(prompt)
            
            print("\n" + "="*60)
            print("GENERATED TEXT:")
            print("="*60)
            print(generated_text)
            print("\n" + "="*60)
            print("PERFORMANCE METRICS:")
            print("="*60)
            print(f"End-to-end latency: {metrics.end_to_end_latency:.3f}s")
            print(f"Token acceptance rate: {metrics.token_acceptance_rate:.1%}")
            print(f"Total tokens generated: {metrics.total_tokens_generated}")
            print(f"Network latency: {metrics.network_latency:.3f}s")
            print(f"Edge inference time: {metrics.edge_inference_time:.3f}s")
            print("="*60)
    
    except KeyboardInterrupt:
        g_logger.info("Edge client shutdown requested")
    
    finally:
        await edge_client.Disconnect()
        g_logger.info("Edge client stopped")

if __name__ == "__main__":
    print("SpecECD Edge Client")
    print("Usage:")
    print("  python run_edge.py [--host HOST] [--port PORT] [--device {cpu,gpu,npu}] [--model-path PATH]")
    print("  python run_edge.py --list-devices")
    print("")
    print("Examples:")
    print("  python run_edge.py                     # Use default settings")
    print("  python run_edge.py --device cpu        # Force CPU inference")
    print("  python run_edge.py --device gpu        # Force GPU acceleration")
    print("  python run_edge.py --device npu --model-path /path/to/onnx/model  # Force WCR NPU acceleration")
    print("  python run_edge.py --host 192.168.1.100 --port 8765  # Remote cloud")
    print("")
    
    asyncio.run(RunEdgeClient())
