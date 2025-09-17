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
    return parser.parse_args()

if __name__ == "__main__":
    args = ParseArguments()
    
    print("SpecECD Performance Test Suite")
    print("Testing speculative decoding with corrected implementation")
    print("Make sure cloud server is running before starting tests")
    print(f"Testing with {args.num_prompts} prompts, {args.iterations} iterations each")
    if args.device:
        print(f"Edge device override: {args.device}")
    else:
        print("Edge device: using config.toml setting")
    print("")
    
    try:
        asyncio.run(RunPerformanceTest(
            num_prompts=args.num_prompts,
            num_iterations=args.iterations,
            cloud_host=args.host,
            cloud_port=args.port,
            edge_device=args.device
        ))
    except KeyboardInterrupt:
        g_logger.info("Performance test interrupted")
    except Exception as e:
        g_logger.error(f"Test error: {e}")
        sys.exit(1)
