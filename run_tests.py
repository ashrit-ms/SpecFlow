"""
Main entry point for running performance tests
Tests the speculative decoding with performance metrics and comparison
"""
import asyncio
import sys
import argparse
import logging
from tests.performance_test import RunPerformanceTest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
g_logger = logging.getLogger(__name__)

def ParseArguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SpecECD Performance Test Suite')
    parser.add_argument(
        '--num-prompts', 
        type=int, 
        default=5, 
        help='Number of test prompts to use (default: 5, max: 10)'
    )
    parser.add_argument(
        '--iterations', 
        type=int, 
        default=2, 
        help='Number of iterations per prompt (default: 2)'
    )
    parser.add_argument(
        '--host', 
        type=str, 
        default='localhost', 
        help='Cloud server host (default: localhost)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8765, 
        help='Cloud server port (default: 8765)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'gpu', 'npu'],
        help='Edge device to use for testing (cpu, gpu, npu). If not specified, uses config.toml setting'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to ONNX model folder (required for --device npu)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO). Use DEBUG for detailed debugging output'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose debug logging (equivalent to --log-level DEBUG)'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = ParseArguments()
    
    # Configure logging level based on arguments
    log_level = logging.DEBUG if args.verbose else getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Also set the root logger level to ensure all modules use the same level
    logging.getLogger().setLevel(log_level)
    
    print("SpecECD Performance Test Suite")
    print("Testing speculative decoding with corrected implementation")
    print("Make sure cloud server is running before starting tests")
    print(f"Testing with {args.num_prompts} prompts, {args.iterations} iterations each")
    print(f"Logging level: {args.log_level if not args.verbose else 'DEBUG (verbose)'}")
    if args.device:
        print(f"Edge device override: {args.device}")
        if args.model_path:
            print(f"Model path: {args.model_path}")
    else:
        print("Edge device: using config.toml setting")
    print("")
    
    # Validate NPU requirements
    if args.device == 'npu':
        if not args.model_path:
            g_logger.error("--model-path is required when using --device npu")
            g_logger.error("Example: python run_tests.py --device npu --model-path /path/to/onnx/model")
            sys.exit(1)
        
        try:
            from edge.wcr_npu_model import ValidateWCRModelPath
            if not ValidateWCRModelPath(args.model_path):
                g_logger.error(f"Invalid WCR ONNX model path: {args.model_path}")
                g_logger.error("Path must contain ONNX files and config.json")
                sys.exit(1)
        except ImportError:
            g_logger.error("WCR NPU support not available")
            g_logger.error("Install with: pip install -r requirements-edge-wcr-npu.txt")
            sys.exit(1)
    
    try:
        asyncio.run(RunPerformanceTest(
            num_prompts=args.num_prompts,
            num_iterations=args.iterations,
            cloud_host=args.host,
            cloud_port=args.port,
            edge_device=args.device,
            model_path=args.model_path
        ))
    except KeyboardInterrupt:
        g_logger.info("Performance test interrupted")
    except Exception as e:
        g_logger.error(f"Test error: {e}")
        sys.exit(1)
