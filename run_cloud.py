"""
Main entry point for running cloud server
Run this on the cloud machine (desktop)
"""
import asyncio
import sys
import logging
from cloud.server import RunCloudServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
g_logger = logging.getLogger(__name__)

if __name__ == "__main__":
    print("SpecECD Cloud Server")
    print("Starting cloud server for speculative decoding...")
    print("Press Ctrl+C to stop the server")
    print("")
    
    try:
        asyncio.run(RunCloudServer())
    except KeyboardInterrupt:
        g_logger.info("Cloud server stopped")
    except Exception as e:
        g_logger.error(f"Server error: {e}")
        sys.exit(1)
