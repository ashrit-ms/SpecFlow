"""
Interactive Speculative Decoding Demo
Shows real-time edge-cloud collaboration with visual feedback
"""
import asyncio
import sys
import time
import argparse
import logging
from typing import List, Tuple
from pathlib import Path

# Color formatting for terminal output
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    # Fallback if colorama not available
    COLORS_AVAILABLE = False
    class Fore:
        GREEN = ""
        RED = ""
        BLUE = ""
        CYAN = ""
        YELLOW = ""
        RESET = ""
    class Style:
        BRIGHT = ""
        RESET_ALL = ""

# Enable ANSI escape sequences in Windows Command Prompt/PowerShell
import os
if os.name == 'nt':
    try:
        # Enable ANSI escape sequences in Windows 10/11 Command Prompt/PowerShell
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except:
        pass  # Ignore if it fails

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from edge.client import EdgeClient
from common.protocol import PerformanceMetrics

# Suppress logging for clean demo output
logging.getLogger().setLevel(logging.ERROR)

class SpeculativeDemo:
    """Interactive demo for speculative decoding visualization"""
    
    def __init__(self, cloud_host: str = "localhost", cloud_port: int = 8765, 
                 edge_device: str = None, model_path: str = None):
        self.cloud_host = cloud_host
        self.cloud_port = cloud_port
        self.edge_device = edge_device
        self.model_path = model_path
        self.edge_client = None
        
        # Demo statistics
        self.total_prompts = 0
        self.total_tokens_generated = 0
        self.total_tokens_accepted = 0
        self.total_tokens_rejected = 0
        self.total_time = 0.0
        
        # Display buffer - contains all tokens to be shown after "Answer: "
        self.buffer = []  # List of tokens that will be displayed
        self.edge_start_index = 0  # Track where current edge tokens start in buffer
        self.current_prompt = ""  # Store current prompt for refresh
        
    async def initialize(self) -> bool:
        """Initialize the edge client and connect to cloud"""
        print(f"{Fore.CYAN}🔧 Initializing Speculative Decoding Demo...")
        
        # Initialize edge client
        self.edge_client = EdgeClient(
            self.cloud_host, 
            self.cloud_port, 
            self.edge_device, 
            self.model_path
        )
        
        if not await self.edge_client.Initialize():
            print(f"{Fore.RED}❌ Failed to initialize edge client")
            return False
        
        if not await self.edge_client.ConnectToCloud():
            print(f"{Fore.RED}❌ Failed to connect to cloud server at {self.cloud_host}:{self.cloud_port}")
            print("   Make sure the cloud server is running:")
            print("   python run_cloud.py")
            return False
        
        print(f"{Fore.GREEN}✅ Connected to cloud server")
        print(f"{Fore.GREEN}✅ Edge model ready")
        print()
        return True
    
    def format_token(self, token: str, accepted: bool) -> str:
        """Format a token with appropriate color coding"""
        if not COLORS_AVAILABLE:
            return f"[REJECTED]{token}" if not accepted else token
        
        if accepted:
            return f"{Fore.GREEN}{token}{Style.RESET_ALL}"
        else:
            return f"{Fore.RED}{token}{Style.RESET_ALL}"
    
    def refresh(self):
        """Refresh the answer content by clearing screen and reprinting everything"""
        import os
        import sys
        
        # Clear screen - use Windows-specific method for PowerShell
        if os.name == 'nt':  # Windows
            os.system('cls')
        else:  # Unix/Linux/Mac
            os.system('clear')
        
        # Reprint the prompt and answer header
        print(f"{Fore.BLUE}Prompt: {self.current_prompt}")
        print(f"{Fore.YELLOW}Answer:")
        
        # Display all tokens from buffer, replacing newlines with spaces for single-line display
        for token in self.buffer:
            # Replace newlines with spaces to keep everything on one line
            clean_token = token.replace('\n', ' ')
            print(self.format_token(clean_token, True), end="", flush=True)
    
    def add_edge_tokens(self, tokens: List[str]):
        """Add edge inference tokens to buffer and refresh display"""
        # Remember where these edge tokens start in the buffer
        self.edge_start_index = len(self.buffer)
        
        # Add edge tokens to buffer
        self.buffer.extend(tokens)
        
        # Refresh display
        self.refresh()
    
    def update_edge_tokens(self, corrected_tokens: List[str]):
        """Update the current edge tokens in buffer based on cloud corrections and refresh"""
        # Get the current edge tokens for comparison
        current_edge_tokens = self.buffer[self.edge_start_index:]
        
        # Show rejected tokens in red if there are any
        corrected_count = min(len(corrected_tokens), len(current_edge_tokens))
        
        # Find tokens that are being rejected/changed
        for i in range(len(current_edge_tokens)):
            if i >= corrected_count or (i < len(corrected_tokens) and current_edge_tokens[i] != corrected_tokens[i]):
                # This token is being rejected - show it in red
                self.buffer[self.edge_start_index + i] = self.format_token(current_edge_tokens[i], False)
        
        # If we have rejections, show them briefly
        if len(current_edge_tokens) > corrected_count or any(
            i < len(corrected_tokens) and current_edge_tokens[i] != corrected_tokens[i] 
            for i in range(min(len(current_edge_tokens), len(corrected_tokens)))
        ):
            self.refresh()
            time.sleep(0.3)  # Brief pause to show rejection
        
        # Now replace with corrected tokens (accepted in green)
        del self.buffer[self.edge_start_index:]
        for token in corrected_tokens:
            self.buffer.append(self.format_token(token, True))
        
        # Refresh display
        self.refresh()
    
    async def run_speculative_generation(self, prompt: str, max_tokens: int = None) -> Tuple[str, PerformanceMetrics]:
        """Run speculative generation with buffer-based display updates"""
        # Store prompt for refresh
        self.current_prompt = prompt
        
        # Print initial display
        print(f"{Fore.BLUE}Prompt: {prompt}")
        print(f"{Fore.YELLOW}Answer:")
        
        # Reset buffer for this new prompt
        self.buffer = []
        self.edge_start_index = 0
        
        # Track token-level statistics for this generation
        generation_accepted = 0
        generation_rejected = 0
        generated_text = ""
        current_prompt = prompt
        total_tokens_generated = 0  # Track total tokens for demo limit
        
        start_time = time.time()
        
        # Real-time speculative decoding loop - continue until model stops naturally or demo limit
        while total_tokens_generated < 100:  # Limit demo to 100 tokens
            # Generate draft tokens on edge
            draft_tokens, draft_probs, edge_inference_time = self.edge_client.m_draft_model.GenerateDraftTokensWithProbabilities(
                current_prompt, 
                5  # Always generate up to 5 draft tokens
            )
            
            if not draft_tokens:
                print(f"\n{Fore.YELLOW}Edge model stopped generating tokens.")
                break
            
            # STEP 1: Add edge tokens to buffer and refresh display
            self.add_edge_tokens(draft_tokens)
            
            # Create verification request
            from common.protocol import SpeculativeRequest, SerializeMessage, DeserializeMessage, SpeculativeResponse, CreateTimestamp
            import asyncio
            
            request = SpeculativeRequest(
                prompt=current_prompt,
                draft_tokens=draft_tokens,
                draft_probabilities=draft_probs,
                max_new_tokens=1000,  # Large limit instead of None to avoid server comparison issues
                request_id=f"req_{CreateTimestamp()}",
                timestamp=CreateTimestamp()
            )
            
            # Send to cloud for verification
            try:
                await self.edge_client.m_websocket.send(SerializeMessage(request))
                response_data = await asyncio.wait_for(
                    self.edge_client.m_websocket.recv(),
                    timeout=30.0
                )
                response = DeserializeMessage(response_data, SpeculativeResponse)
            except Exception as e:
                print(f"\n{Fore.RED}Cloud communication error: {e}")
                print(f"{Fore.YELLOW}Falling back to edge-only mode...")
                
                # Fallback: Just use the draft tokens as-is when cloud is unavailable
                class FallbackResponse:
                    def __init__(self):
                        self.accepted_count = len(draft_tokens)  # Accept all draft tokens
                        self.new_tokens = []  # No additional tokens
                        self.verified_tokens = draft_tokens
                        self.early_exit = False
                
                response = FallbackResponse()
            
            # STEP 2: Create corrected token list and update buffer
            verified_count = response.accepted_count
            
            # Build corrected tokens: accepted edge tokens + new cloud tokens
            corrected_tokens = []
            
            # Add accepted edge tokens
            if verified_count > 0:
                corrected_tokens.extend(draft_tokens[:verified_count])
            
            # Add new cloud tokens
            corrected_tokens.extend(response.new_tokens)
            
            # Update buffer with corrected tokens and refresh
            self.update_edge_tokens(corrected_tokens)
            
            # Update statistics and generated text
            generation_accepted += verified_count + len(response.new_tokens)
            generation_rejected += len(draft_tokens) - verified_count
            total_tokens_generated += len(corrected_tokens)  # Update demo token counter
            
            # Add corrected tokens to generated text
            for token in corrected_tokens:
                generated_text += token
            
            # Update for next iteration
            accepted_text = "".join(response.verified_tokens) + "".join(response.new_tokens)
            current_prompt += accepted_text
            
            # Check for early exit, end-of-sequence, or demo limit reached
            if response.early_exit:
                break
            elif not corrected_tokens:
                break
            elif total_tokens_generated >= 100:
                print(f"\n{Fore.YELLOW}Demo limit reached (100 tokens).")
                break
        
        print()  # New line after answer
        
        # Calculate metrics
        total_time = time.time() - start_time
        metrics = PerformanceMetrics()
        metrics.end_to_end_latency = total_time
        metrics.token_acceptance_rate = generation_accepted / (generation_accepted + generation_rejected) if (generation_accepted + generation_rejected) > 0 else 0
        
        # Update statistics
        self.total_tokens_generated += (generation_accepted + generation_rejected)
        self.total_tokens_accepted += generation_accepted
        self.total_tokens_rejected += generation_rejected
        self.total_time += total_time
        
        return generated_text, metrics
    
    def _generate_plausible_wrong_token(self, correct_token: str) -> str:
        """Generate a plausible wrong token for demo purposes"""
        # Simple substitutions to simulate edge model mistakes
        substitutions = {
            "the": "a",
            "a": "the", 
            "is": "was",
            "was": "is",
            "are": "were",
            "were": "are",
            "life": "existence",
            "meaning": "purpose",
            "question": "topic",
            "complex": "difficult",
            "philosophy": "thinking",
            "human": "people",
            "understanding": "knowledge",
            "important": "significant",
            "different": "various",
            "many": "numerous"
        }
        
        return substitutions.get(correct_token.lower(), correct_token + "s")
    
    def show_statistics(self):
        """Display generation statistics"""
        if self.total_prompts == 0:
            return
        
        print(f"\n{Fore.CYAN}{'='*50}")
        print(f"{Fore.CYAN}📊 Speculative Decoding Statistics")
        print(f"{Fore.CYAN}{'='*50}")
        
        acceptance_rate = (self.total_tokens_accepted / self.total_tokens_generated) * 100 if self.total_tokens_generated > 0 else 0
        rejection_rate = (self.total_tokens_rejected / self.total_tokens_generated) * 100 if self.total_tokens_generated > 0 else 0
        avg_time = self.total_time / self.total_prompts if self.total_prompts > 0 else 0
        
        print(f"{Fore.GREEN}✅ Tokens Accepted: {self.total_tokens_accepted} ({acceptance_rate:.1f}%)")
        print(f"{Fore.RED}❌ Tokens Rejected: {self.total_tokens_rejected} ({rejection_rate:.1f}%)")
        print(f"{Fore.BLUE}📝 Total Tokens: {self.total_tokens_generated}")
        print(f"{Fore.YELLOW}⏱️  Average Time: {avg_time:.2f}s per prompt")
        print(f"{Fore.CYAN}🔄 Total Prompts: {self.total_prompts}")
        
        if acceptance_rate > 70:
            print(f"{Fore.GREEN}🎉 Excellent edge-cloud alignment!")
        elif acceptance_rate > 50:
            print(f"{Fore.YELLOW}👍 Good speculative decoding performance")
        else:
            print(f"{Fore.RED}⚠️  Edge model could benefit from better alignment")
    
    async def run_interactive_demo(self):
        """Run the interactive demo loop"""
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}")
        print(f"{Fore.CYAN}{Style.BRIGHT}🚀 Interactive Speculative Decoding Demo")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}")
        print(f"{Fore.BLUE}Edge Device: {self.edge_device or 'config default'}")
        if self.model_path:
            print(f"{Fore.BLUE}Model Path: {self.model_path}")
        print(f"{Fore.BLUE}Cloud Server: {self.cloud_host}:{self.cloud_port}")
        print()
        print(f"{Fore.YELLOW}Legend:")
        print(f"  {Fore.GREEN}Green text{Style.RESET_ALL} = Accepted edge tokens")
        print(f"  {Fore.RED}Red text{Style.RESET_ALL} = Rejected edge tokens")
        print()
        
        # Predefined demo prompts
        demo_prompts = [
            "What is the meaning of life?",
            "How does artificial intelligence work?",
            "What are the benefits of renewable energy?",
            "Explain quantum computing in simple terms.",
            "What is the future of space exploration?"
        ]
        
        while True:
            print(f"{Fore.CYAN}{'='*50}")
            print(f"{Fore.CYAN}Choose an option:")
            print(f"{Fore.YELLOW}1. Quick demo prompts")
            print(f"{Fore.YELLOW}2. Enter custom prompt")
            print(f"{Fore.YELLOW}3. Show statistics")
            print(f"{Fore.YELLOW}4. Exit")
            print()
            
            try:
                choice = input(f"{Fore.BLUE}Enter choice (1-4): {Style.RESET_ALL}").strip()
                
                if choice == "1":
                    print(f"\n{Fore.CYAN}Demo Prompts:")
                    for i, prompt in enumerate(demo_prompts, 1):
                        print(f"{Fore.YELLOW}{i}. {prompt}")
                    
                    try:
                        prompt_choice = int(input(f"{Fore.BLUE}Select prompt (1-{len(demo_prompts)}): {Style.RESET_ALL}"))
                        if 1 <= prompt_choice <= len(demo_prompts):
                            selected_prompt = demo_prompts[prompt_choice - 1]
                            print()
                            await self.run_speculative_generation(selected_prompt)
                            self.total_prompts += 1
                        else:
                            print(f"{Fore.RED}Invalid choice!")
                    except ValueError:
                        print(f"{Fore.RED}Please enter a valid number!")
                
                elif choice == "2":
                    custom_prompt = input(f"{Fore.BLUE}Enter your prompt: {Style.RESET_ALL}").strip()
                    if custom_prompt:
                        print()
                        await self.run_speculative_generation(custom_prompt)
                        self.total_prompts += 1
                    else:
                        print(f"{Fore.RED}Please enter a valid prompt!")
                
                elif choice == "3":
                    self.show_statistics()
                
                elif choice == "4":
                    print(f"{Fore.CYAN}Thank you for using the Speculative Decoding Demo!")
                    self.show_statistics()
                    break
                
                else:
                    print(f"{Fore.RED}Invalid choice! Please enter 1-4.")
                
                print()
                
            except KeyboardInterrupt:
                print(f"\n{Fore.CYAN}Demo interrupted. Goodbye!")
                self.show_statistics()
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {e}")
                continue
    
    async def cleanup(self):
        """Clean up resources"""
        if self.edge_client:
            await self.edge_client.Disconnect()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Interactive Speculative Decoding Demo')
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
        help='Edge device to use (cpu, gpu, npu). If not specified, uses config.toml setting'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to ONNX model folder (required for --device npu)'
    )
    return parser.parse_args()

async def main():
    """Main demo function"""
    args = parse_arguments()
    
    # Validate NPU requirements
    if args.device == 'npu':
        if not args.model_path:
            print(f"{Fore.RED}❌ --model-path is required when using --device npu")
            print("   Example: python demo_speculative.py --device npu --model-path /path/to/onnx/model")
            sys.exit(1)
        
        try:
            from edge.wcr_npu_model import ValidateWCRModelPath
            if not ValidateWCRModelPath(args.model_path):
                print(f"{Fore.RED}❌ Invalid WCR ONNX model path: {args.model_path}")
                print("   Path must contain ONNX files and config.json")
                sys.exit(1)
        except ImportError:
            print(f"{Fore.RED}❌ WCR NPU support not available")
            print("   Install with: pip install -r requirements-edge-wcr-npu.txt")
            sys.exit(1)
    
    # Create and run demo
    demo = SpeculativeDemo(
        cloud_host=args.host,
        cloud_port=args.port,
        edge_device=args.device,
        model_path=args.model_path
    )
    
    try:
        if await demo.initialize():
            await demo.run_interactive_demo()
    except KeyboardInterrupt:
        print(f"\n{Fore.CYAN}Demo interrupted.")
    except Exception as e:
        print(f"{Fore.RED}Demo error: {e}")
    finally:
        await demo.cleanup()

if __name__ == "__main__":
    # Install colorama if not available
    try:
        import colorama
    except ImportError:
        print("Installing colorama for colored output...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "colorama"])
        import colorama
    
    asyncio.run(main())