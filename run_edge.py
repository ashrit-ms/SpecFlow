"""
Main entry point for running edge client
Run this on the edge machine (laptop)
"""
import asyncio
import sys
import logging
from edge.client import EdgeClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
g_logger = logging.getLogger(__name__)

async def RunEdgeClient():
    """Main function to run edge client"""
    # Parse command line arguments
    cloud_host = "localhost"  # Change to desktop IP address
    cloud_port = 8765
    
    if len(sys.argv) > 1:
        cloud_host = sys.argv[1]
    if len(sys.argv) > 2:
        cloud_port = int(sys.argv[2])
    
    g_logger.info(f"Starting edge client, connecting to {cloud_host}:{cloud_port}")
    
    # Create and initialize edge client
    edge_client = EdgeClient(cloud_host, cloud_port)
    
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
    print("Usage: python run_edge.py [cloud_host] [cloud_port]")
    print(f"Default: python run_edge.py localhost 8765")
    print("")
    
    asyncio.run(RunEdgeClient())
