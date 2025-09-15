"""
Cloud server for handling edge requests and serving target model
WebSocket server that processes verification requests from edge clients
"""
import asyncio
import websockets
import json
import logging
from typing import Set

from common.protocol import (
    SpeculativeRequest, SpeculativeResponse, BaselineRequest, BaselineResponse,
    SerializeMessage, DeserializeMessage, CreateTimestamp,
    CalculateAcceptanceRate
)
from cloud.target_model import CloudTargetModel

# Configure logging
logging.basicConfig(level=logging.INFO)
g_logger = logging.getLogger(__name__)

class CloudServer:
    """WebSocket server running target model for verification"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.m_host = host
        self.m_port = port
        self.m_target_model = CloudTargetModel()
        self.m_connected_clients = set()
        
    async def Initialize(self) -> bool:
        """Initialize the cloud server and load target model"""
        g_logger.info("Initializing cloud server...")
        
        # Load target model
        if not self.m_target_model.LoadModel():
            g_logger.error("Failed to load target model")
            return False
            
        g_logger.info("Cloud server initialized successfully")
        return True
    
    async def HandleClient(self, websocket):
        """Handle incoming client connections"""
        self.m_connected_clients.add(websocket)
        client_address = websocket.remote_address
        g_logger.info(f"Edge client connected: {client_address}")
        
        try:
            async for message in websocket:
                await self._ProcessRequest(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            g_logger.info(f"Edge client disconnected: {client_address}")
        except Exception as e:
            g_logger.error(f"Error handling client {client_address}: {e}")
        finally:
            self.m_connected_clients.discard(websocket)
    
    async def _ProcessRequest(self, websocket, message: str):
        """Process verification or baseline request from edge client"""
        try:
            # Try to determine request type by parsing JSON first
            import json
            data = json.loads(message)
            
            # Check if this is a baseline request or speculative request
            if 'draft_tokens' in data:
                # This is a speculative decoding request
                await self._ProcessSpeculativeRequest(websocket, data)
            else:
                # This is a baseline request
                await self._ProcessBaselineRequest(websocket, data)
                
        except Exception as e:
            g_logger.error(f"Failed to process request: {e}")
            # Send error response
            error_response = SpeculativeResponse(
                verified_tokens=[],
                new_tokens=[],
                accepted_count=0,
                total_draft_count=0,
                early_exit=True,
                request_id="error",
                timestamp=CreateTimestamp()
            )
            await websocket.send(SerializeMessage(error_response))
    
    async def _ProcessSpeculativeRequest(self, websocket, data: dict):
        """Process speculative decoding request"""
        try:
            # Parse as speculative request
            request = SpeculativeRequest(**data)
            g_logger.info(f"Processing speculative request {request.request_id} with {len(request.draft_tokens)} draft tokens")
            
            # Verify draft tokens and generate additional tokens using probabilistic method
            start_time = CreateTimestamp()
            # Use probabilistic verification if draft probabilities are available
            if request.draft_probabilities is not None and len(request.draft_probabilities) > 0:
                verified_tokens, new_tokens, accepted_count, inference_time = self.m_target_model.VerifyAndCompleteProbabilistic(
                    request.prompt,
                    request.draft_tokens,
                    request.draft_probabilities,  # Use the probabilities from draft model
                    max_new_tokens=min(10, request.max_new_tokens)
                )
                g_logger.info(f"Used probabilistic verification with {len(request.draft_probabilities)} probabilities")
            else:
                # Fallback to legacy method for compatibility
                verified_tokens, new_tokens, accepted_count, inference_time = self.m_target_model.VerifyAndComplete(
                    request.prompt,
                    request.draft_tokens,
                    max_new_tokens=min(10, request.max_new_tokens)
                )
                g_logger.warning("Using legacy string-based verification (no probabilities provided)")
            
            # Create response
            response = SpeculativeResponse(
                verified_tokens=verified_tokens,
                new_tokens=new_tokens,
                accepted_count=accepted_count,
                total_draft_count=len(request.draft_tokens),
                early_exit=False,  # Could implement early exit logic here
                request_id=request.request_id,
                timestamp=CreateTimestamp()
            )
            
            # Send response back to edge
            await websocket.send(SerializeMessage(response))
            
            # Log performance
            total_time = CreateTimestamp() - start_time
            acceptance_rate = CalculateAcceptanceRate(accepted_count, len(request.draft_tokens))
            
            g_logger.info(f"Speculative request {request.request_id} processed in {total_time:.3f}s")
            g_logger.info(f"Acceptance rate: {acceptance_rate:.2%}")
            
        except Exception as e:
            g_logger.error(f"Failed to process speculative request: {e}")
            raise
    
    async def _ProcessBaselineRequest(self, websocket, data: dict):
        """Process baseline (cloud-only) request"""
        try:
            # Parse as baseline request
            request = BaselineRequest(**data)
            g_logger.info(f"Processing baseline request {request.request_id}")
            
            # Generate text using cloud model only
            start_time = CreateTimestamp()
            generated_text = self.m_target_model._GenerateTargetText(
                request.prompt, 
                request.max_new_tokens
            )
            inference_time = CreateTimestamp() - start_time
            
            # Count tokens
            tokens_generated = len(generated_text.split()) if generated_text else 0
            
            # Create response
            response = BaselineResponse(
                generated_text=generated_text,
                tokens_generated=tokens_generated,
                inference_time=inference_time,
                request_id=request.request_id,
                timestamp=CreateTimestamp()
            )
            
            # Send response back to edge
            await websocket.send(SerializeMessage(response))
            
            # Log performance
            g_logger.info(f"Baseline request {request.request_id} processed in {inference_time:.3f}s")
            g_logger.info(f"Generated {tokens_generated} tokens")
            
        except Exception as e:
            g_logger.error(f"Failed to process baseline request: {e}")
            raise
    
    async def StartServer(self):
        """Start the WebSocket server"""
        g_logger.info(f"Starting cloud server on {self.m_host}:{self.m_port}")
        
        try:
            server = await websockets.serve(
                self.HandleClient,
                self.m_host,
                self.m_port,
                ping_interval=300,  # 5 minutes - match client settings
                ping_timeout=300,   # 5 minutes - match client settings
                close_timeout=60    # 1 minute close timeout
            )
            
            g_logger.info("Cloud server started successfully")
            g_logger.info("Waiting for edge client connections...")
            
            # Keep server running
            await server.wait_closed()
            
        except Exception as e:
            g_logger.error(f"Failed to start server: {e}")
            raise
    
    def GetServerInfo(self) -> dict:
        """Get information about the server and model"""
        model_info = self.m_target_model.GetModelInfo()
        return {
            "server_host": self.m_host,
            "server_port": self.m_port,
            "connected_clients": len(self.m_connected_clients),
            "model_info": model_info
        }

async def RunCloudServer():
    """Main function to run the cloud server"""
    server = CloudServer()
    
    # Initialize server
    if not await server.Initialize():
        g_logger.error("Failed to initialize cloud server")
        return
    
    # Print server information
    info = server.GetServerInfo()
    g_logger.info(f"Server info: {info}")
    
    # Start server
    try:
        await server.StartServer()
    except KeyboardInterrupt:
        g_logger.info("Server shutdown requested")
    except Exception as e:
        g_logger.error(f"Server error: {e}")

if __name__ == "__main__":
    # Run the cloud server
    asyncio.run(RunCloudServer())
