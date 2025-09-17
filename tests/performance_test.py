"""
Test suite for evaluating speculative edge-cloud decoding performance
Tests the corrected implementation and measures key performance metrics
"""
import asyncio
import time
import statistics
import sys
import os
from typing import List, Dict, Any
import logging
from dataclasses import dataclass
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from edge.client import EdgeClient
from common.protocol import PerformanceMetrics, CreateTimestamp, BaselineRequest, BaselineResponse, SerializeMessage, DeserializeMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
g_logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Results from a single test run"""
    prompt: str
    generated_text: str
    metrics: PerformanceMetrics
    success: bool = True
    error_message: str = ""

class SpecECDTestSuite:
    """Test suite for evaluating SpecECD performance with corrected implementation"""
    
    def __init__(self, cloud_host: str = "localhost", cloud_port: int = 8765, edge_device: str = None):
        self.m_cloud_host = cloud_host
        self.m_cloud_port = cloud_port
        self.m_edge_device = edge_device  # Device override for testing
        self.m_test_prompts = self._CreateTestPrompts()
        self.m_results: List[TestResult] = []
        
    def _CreateTestPrompts(self) -> List[str]:
        """Create diverse test prompts for language generation"""
        return [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly",
            "Scientists have recently discovered that",
            "The impact of climate change on",
            "Machine learning algorithms can help us",
            "Write a C++ function that prints 'Hello, World!'",
            "Explain how quantum computing works:",
            "The benefits of renewable energy include",
            "In the field of medicine, new research shows",
            "The evolution of programming languages has"
        ]
    
    async def RunTestSuite(self, num_prompts: int = 5, num_iterations: int = 2) -> Dict[str, Any]:
        """
        Run the complete test suite with corrected speculative decoding
        Returns aggregated results and performance metrics
        """
        results, connection = await self.RunTestSuiteWithConnection(num_prompts, num_iterations)
        if connection:
            await connection.close()
        return results
    
    async def RunTestSuiteWithConnection(self, num_prompts: int = 5, num_iterations: int = 2):
        """
        Run the test suite and return both results and the connection for reuse
        Returns (results, websocket_connection)
        """
        # Limit prompts to available ones
        num_prompts = min(num_prompts, len(self.m_test_prompts))
        selected_prompts = self.m_test_prompts[:num_prompts]
        
        g_logger.info(f"Starting SpecECD test suite with {num_prompts} prompts")
        g_logger.info(f"Running {num_iterations} iterations per prompt")
        
        return await self._RunAsyncTestSuiteWithConnection(selected_prompts, num_iterations)
    
    async def _RunAsyncTestSuiteWithConnection(self, prompts: List[str], num_iterations: int):
        """Run the async version of the test suite and return the connection"""
        # Initialize edge client with device override if specified
        edge_client = EdgeClient(self.m_cloud_host, self.m_cloud_port, self.m_edge_device)
        
        if not await edge_client.Initialize():
            g_logger.error("Failed to initialize edge client")
            return {"error": "Failed to initialize edge client"}, None
        
        if not await edge_client.ConnectToCloud():
            g_logger.error("Failed to connect to cloud server")
            return {"error": "Failed to connect to cloud server"}, None
        
        # Run tests
        total_tests = len(prompts) * num_iterations
        test_count = 0
        
        try:
            for iteration in range(num_iterations):
                g_logger.info(f"Running iteration {iteration + 1}/{num_iterations}")
                
                for i, prompt in enumerate(prompts):
                    test_count += 1
                    g_logger.info(f"Test {test_count}/{total_tests}: Running prompt {i + 1}")
                    
                    # Run test
                    result = await self._RunSingleTest(edge_client, prompt)
                    self.m_results.append(result)
                    
                    # Log result
                    if result.success:
                        g_logger.info(f"[OK] Test completed in {result.metrics.end_to_end_latency:.3f}s")
                        g_logger.info(f"[METRICS] Acceptance rate: {result.metrics.token_acceptance_rate:.2%}")
                    else:
                        g_logger.warning(f"[FAIL] Test failed: {result.error_message}")
                    
                    # Brief pause between tests
                    await asyncio.sleep(1)
            
            # Analyze results but keep the connection open
            results = self._AnalyzeResults()
            return results, edge_client.m_websocket
            
        except Exception as e:
            g_logger.error(f"Test suite failed: {e}")
            await edge_client.Disconnect()
            return {"error": str(e)}, None
    
    async def _RunAsyncTestSuite(self, prompts: List[str], num_iterations: int) -> Dict[str, Any]:
        """Run the async version of the test suite"""
        # Initialize edge client with device override if specified
        edge_client = EdgeClient(self.m_cloud_host, self.m_cloud_port, self.m_edge_device)
        
        if not await edge_client.Initialize():
            g_logger.error("Failed to initialize edge client")
            return {"error": "Failed to initialize edge client"}
        
        if not await edge_client.ConnectToCloud():
            g_logger.error("Failed to connect to cloud server")
            return {"error": "Failed to connect to cloud server"}
        
        # Run tests
        total_tests = len(prompts) * num_iterations
        test_count = 0
        
        try:
            for iteration in range(num_iterations):
                g_logger.info(f"Running iteration {iteration + 1}/{num_iterations}")
                
                for i, prompt in enumerate(prompts):
                    test_count += 1
                    g_logger.info(f"Test {test_count}/{total_tests}: Running prompt {i + 1}")
                    
                    # Run test
                    result = await self._RunSingleTest(edge_client, prompt)
                    self.m_results.append(result)
                    
                    # Log result
                    if result.success:
                        g_logger.info(f"[OK] Test completed in {result.metrics.end_to_end_latency:.3f}s")
                        g_logger.info(f"[METRICS] Acceptance rate: {result.metrics.token_acceptance_rate:.2%}")
                    else:
                        g_logger.warning(f"[FAIL] Test failed: {result.error_message}")
                    
                    # Brief pause between tests
                    await asyncio.sleep(1)
        
        finally:
            await edge_client.Disconnect()
        
        # Analyze results
        return self._AnalyzeResults()
    
    async def _RunSingleTest(self, edge_client: EdgeClient, prompt: str) -> TestResult:
        """Run a single test with the given prompt using corrected implementation"""
        try:
            start_time = CreateTimestamp()
            
            # Generate text using corrected speculative decoding
            generated_text, metrics = await edge_client.GenerateWithSpeculation(
                prompt, 
                max_tokens=50  # Reasonable length for testing
            )
            
            # Create test result
            result = TestResult(
                prompt=prompt,
                generated_text=generated_text,
                metrics=metrics,
                success=bool(generated_text.strip())
            )
            
            if not result.success:
                result.error_message = "No text generated"
            
            return result
            
        except Exception as e:
            g_logger.error(f"Test execution failed: {e}")
            return TestResult(
                prompt=prompt,
                generated_text="",
                metrics=PerformanceMetrics(),
                success=False,
                error_message=str(e)
            )
    
    def _AnalyzeResults(self) -> Dict[str, Any]:
        """Analyze test results and calculate aggregate metrics"""
        if not self.m_results:
            return {"error": "No test results to analyze"}
        
        successful_results = [r for r in self.m_results if r.success]
        
        if not successful_results:
            return {"error": "No successful test results"}
        
        # Calculate aggregate metrics
        latencies = [r.metrics.end_to_end_latency for r in successful_results]
        acceptance_rates = [r.metrics.token_acceptance_rate for r in successful_results]
        tokens_generated = [r.metrics.total_tokens_generated for r in successful_results]
        network_latencies = [r.metrics.network_latency for r in successful_results]
        edge_inference_times = [r.metrics.edge_inference_time for r in successful_results]
        
        # Calculate statistics
        stats = {
            "total_tests": len(self.m_results),
            "successful_tests": len(successful_results),
            "success_rate": len(successful_results) / len(self.m_results),
            
            "latency_stats": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0
            },
            
            "acceptance_rate_stats": {
                "mean": statistics.mean(acceptance_rates),
                "median": statistics.median(acceptance_rates),
                "min": min(acceptance_rates),
                "max": max(acceptance_rates),
                "std_dev": statistics.stdev(acceptance_rates) if len(acceptance_rates) > 1 else 0
            },
            
            "tokens_generated_stats": {
                "mean": statistics.mean(tokens_generated),
                "median": statistics.median(tokens_generated),
                "total": sum(tokens_generated)
            },
            
            "network_latency_stats": {
                "mean": statistics.mean(network_latencies),
                "median": statistics.median(network_latencies)
            },
            
            "edge_inference_stats": {
                "mean": statistics.mean(edge_inference_times),
                "median": statistics.median(edge_inference_times)
            }
        }
        
        # Sample generated text
        stats["sample_results"] = [
            {
                "prompt": r.prompt[:100] + "..." if len(r.prompt) > 100 else r.prompt,
                "generated_text": r.generated_text[:200] + "..." if len(r.generated_text) > 200 else r.generated_text,
                "latency": r.metrics.end_to_end_latency,
                "acceptance_rate": r.metrics.token_acceptance_rate
            }
            for r in successful_results[:3]  # Show first 3 successful results
        ]
        
        return stats
    
    def GenerateReport(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable test report"""
        if "error" in results:
            return f"Test Report - ERROR: {results['error']}"
        
        report = []
        report.append("=" * 60)
        report.append("SpecECD Performance Test Report")
        report.append("Corrected Implementation with Probabilistic Verification")
        report.append("=" * 60)
        report.append("")
        
        # Test summary
        report.append(f"Total Tests: {results['total_tests']}")
        report.append(f"Successful Tests: {results['successful_tests']}")
        report.append(f"Success Rate: {results['success_rate']:.1%}")
        report.append("")
        
        # Latency metrics
        lat_stats = results['latency_stats']
        report.append("End-to-End Latency:")
        report.append(f"  Mean: {lat_stats['mean']:.3f}s")
        report.append(f"  Median: {lat_stats['median']:.3f}s")
        report.append(f"  Min/Max: {lat_stats['min']:.3f}s / {lat_stats['max']:.3f}s")
        report.append(f"  Std Dev: {lat_stats['std_dev']:.3f}s")
        report.append("")
        
        # Acceptance rate metrics (KEY IMPROVEMENT)
        acc_stats = results['acceptance_rate_stats']
        report.append("Token Acceptance Rate (CORRECTED ALGORITHM):")
        report.append(f"  Mean: {acc_stats['mean']:.1%}")
        report.append(f"  Median: {acc_stats['median']:.1%}")
        report.append(f"  Min/Max: {acc_stats['min']:.1%} / {acc_stats['max']:.1%}")
        report.append("")
        
        # Performance assessment
        mean_acceptance = acc_stats['mean']
        if mean_acceptance > 0.5:
            assessment = "EXCELLENT - Algorithm working correctly"
        elif mean_acceptance > 0.3:
            assessment = "GOOD - Significant improvement achieved"
        elif mean_acceptance > 0.1:
            assessment = "MODERATE - Some improvement observed"
        else:
            assessment = "POOR - Algorithm may need further tuning"
        
        report.append(f"Performance Assessment: {assessment}")
        report.append("")
        
        # Token generation
        token_stats = results['tokens_generated_stats']
        report.append("Tokens Generated:")
        report.append(f"  Mean per test: {token_stats['mean']:.1f}")
        report.append(f"  Total: {token_stats['total']}")
        report.append("")
        
        # Network performance
        net_stats = results['network_latency_stats']
        edge_stats = results['edge_inference_stats']
        report.append("Component Performance:")
        report.append(f"  Network Latency (mean): {net_stats['mean']:.3f}s")
        report.append(f"  Edge Inference (mean): {edge_stats['mean']:.3f}s")
        report.append("")
        
        # Sample results
        report.append("Sample Results:")
        for i, sample in enumerate(results['sample_results'], 1):
            report.append(f"  Test {i}:")
            report.append(f"    Prompt: {sample['prompt']}")
            report.append(f"    Generated: {sample['generated_text']}")
            report.append(f"    Latency: {sample['latency']:.3f}s")
            report.append(f"    Acceptance Rate: {sample['acceptance_rate']:.1%}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)

async def RunPerformanceTest(num_prompts: int = 5, num_iterations: int = 2, 
                           cloud_host: str = "localhost", cloud_port: int = 8765,
                           edge_device: str = None):
    """Main function to run performance tests with corrected implementation and baseline comparison"""
    g_logger.info("Starting SpecECD performance evaluation with corrected algorithm")
    
    if edge_device:
        g_logger.info(f"Edge device override: {edge_device}")
    else:
        g_logger.info("Edge device: using config.toml setting")
    
    # Run speculative decoding test
    g_logger.info("=" * 60)
    g_logger.info("PHASE 1: SPECULATIVE DECODING TEST")
    g_logger.info("=" * 60)
    
    test_suite = SpecECDTestSuite(cloud_host, cloud_port, edge_device)
    speculative_results, shared_connection = await test_suite.RunTestSuiteWithConnection(num_prompts=num_prompts, num_iterations=num_iterations)
    
    # Run cloud-only baseline test using the same connection
    g_logger.info("\n" + "=" * 60)
    g_logger.info("PHASE 2: CLOUD-ONLY BASELINE TEST (SAME CONNECTION)")
    g_logger.info("=" * 60)
    
    baseline_results = await RunCloudOnlyBaseline(
        num_prompts=num_prompts, 
        num_iterations=num_iterations,
        websocket_connection=shared_connection  # Reuse the same connection
    )
    
    # Close the shared connection
    if shared_connection:
        await shared_connection.close()
        g_logger.info("Shared WebSocket connection closed")
    
    # Generate comparative report
    g_logger.info("\n" + "=" * 60)
    g_logger.info("GENERATING COMPARATIVE ANALYSIS")
    g_logger.info("=" * 60)
    
    comparative_report = GenerateComparativeReport(speculative_results, baseline_results)
    print(comparative_report)
    
    # Save results to file
    with open("test_results.txt", "w", encoding='utf-8') as f:
        f.write(comparative_report)
    
    g_logger.info("Comparative test report saved to test_results.txt")

async def RunCloudOnlyBaseline(num_prompts: int = 5, num_iterations: int = 2, 
                              websocket_connection=None, cloud_host: str = "localhost", cloud_port: int = 8765) -> Dict[str, Any]:
    """Run cloud-only baseline test using the provided WebSocket connection or creating a new one"""
    import websockets
    
    g_logger.info("Starting cloud-only baseline evaluation via network")
    
    # Use same test prompts as speculative test
    test_prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly",
        "Scientists have recently discovered that",
        "The impact of climate change on",
        "Machine learning algorithms can help us",
        "Write a C++ function that prints 'Hello, World!'",
        "Explain how quantum computing works:",
        "The benefits of renewable energy include",
        "In the field of medicine, new research shows",
        "The evolution of programming languages has"
    ]
    
    # Limit prompts to available ones
    num_prompts = min(num_prompts, len(test_prompts))
    selected_prompts = test_prompts[:num_prompts]
    
    g_logger.info(f"Running cloud-only baseline with {num_prompts} prompts")
    g_logger.info(f"Running {num_iterations} iterations per prompt")
    
    # Use provided connection or create new one
    websocket = websocket_connection
    should_close_connection = False
    
    if websocket is None:
        # Create new connection if none provided
        try:
            uri = f"ws://{cloud_host}:{cloud_port}"
            g_logger.info(f"Creating new connection to cloud server at {uri} for baseline test")
            
            websocket = await websockets.connect(
                uri,
                ping_interval=300,
                ping_timeout=300,
                close_timeout=60
            )
            should_close_connection = True
            
        except Exception as e:
            g_logger.error(f"Failed to connect to cloud server for baseline test: {e}")
            return {"error": f"Failed to connect to cloud server: {e}"}
    else:
        g_logger.info("Reusing existing WebSocket connection for baseline test")
    
    results = []
    total_tests = num_prompts * num_iterations
    test_count = 0
    
    try:
        for iteration in range(num_iterations):
            g_logger.info(f"Baseline iteration {iteration + 1}/{num_iterations}")
            
            for i, prompt in enumerate(selected_prompts):
                test_count += 1
                g_logger.info(f"Baseline test {test_count}/{total_tests}: Running prompt {i + 1}")
                
                try:
                    # Create baseline request
                    request = BaselineRequest(
                        prompt=prompt,
                        max_new_tokens=50,  # Same as speculative test
                        request_id=f"baseline_req_{CreateTimestamp()}",
                        timestamp=CreateTimestamp()
                    )
                    
                    # Send request and measure network latency
                    network_start = CreateTimestamp()
                    await websocket.send(SerializeMessage(request))
                    
                    # Receive response with timeout
                    response_data = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=300.0
                    )
                    network_time = CreateTimestamp() - network_start
                    
                    # Parse response
                    response = DeserializeMessage(response_data, BaselineResponse)
                    
                    result = {
                        'prompt': prompt,
                        'generated_text': response.generated_text,
                        'latency': network_time,  # Total network latency (includes inference)
                        'inference_time': response.inference_time,  # Pure inference time
                        'network_overhead': network_time - response.inference_time,  # Network overhead
                        'success': bool(response.generated_text.strip()),
                        'tokens_generated': response.tokens_generated
                    }
                    
                    results.append(result)
                    
                    if result['success']:
                        g_logger.info(f"[OK] Baseline completed in {network_time:.3f}s (inference: {response.inference_time:.3f}s)")
                    else:
                        g_logger.warning(f"[FAIL] Baseline failed")
                    
                except Exception as e:
                    g_logger.error(f"Baseline test failed: {e}")
                    results.append({
                        'prompt': prompt,
                        'generated_text': "",
                        'latency': float('inf'),
                        'inference_time': 0.0,
                        'network_overhead': 0.0,
                        'success': False,
                        'tokens_generated': 0
                    })
                
                # Brief pause between tests
                await asyncio.sleep(1)
    
    finally:
        if should_close_connection and websocket:
            await websocket.close()
    
    # Analyze baseline results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        return {"error": "No successful baseline results"}
    
    latencies = [r['latency'] for r in successful_results]
    inference_times = [r['inference_time'] for r in successful_results]
    network_overheads = [r['network_overhead'] for r in successful_results]
    tokens_generated = [r['tokens_generated'] for r in successful_results]
    
    analysis = {
        "total_tests": len(results),
        "successful_tests": len(successful_results),
        "success_rate": len(successful_results) / len(results),
        "latency_stats": {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0
        },
        "inference_time_stats": {
            "mean": statistics.mean(inference_times),
            "median": statistics.median(inference_times),
            "min": min(inference_times),
            "max": max(inference_times)
        },
        "network_overhead_stats": {
            "mean": statistics.mean(network_overheads),
            "median": statistics.median(network_overheads)
        },
        "tokens_generated_stats": {
            "mean": statistics.mean(tokens_generated),
            "median": statistics.median(tokens_generated),
            "total": sum(tokens_generated)
        },
        "sample_results": [
            {
                "prompt": r['prompt'][:100] + "..." if len(r['prompt']) > 100 else r['prompt'],
                "generated_text": r['generated_text'][:200] + "..." if len(r['generated_text']) > 200 else r['generated_text'],
                "latency": r['latency'],
                "inference_time": r['inference_time'],
                "network_overhead": r['network_overhead']
            }
            for r in successful_results[:3]
        ]
    }
    
    return analysis

def GenerateComparativeReport(speculative_results: Dict[str, Any], baseline_results: Dict[str, Any]) -> str:
    """Generate comprehensive comparative report"""
    report = []
    report.append("=" * 80)
    report.append("SpecECD COMPARATIVE PERFORMANCE ANALYSIS")
    report.append("Speculative Decoding vs Cloud-Only Baseline")
    report.append("=" * 80)
    report.append("")
    
    # Check for errors
    if "error" in speculative_results:
        report.append(f"[ERROR] SPECULATIVE TEST ERROR: {speculative_results['error']}")
        return "\n".join(report)
    
    if "error" in baseline_results:
        report.append(f"[ERROR] BASELINE TEST ERROR: {baseline_results['error']}")
        return "\n".join(report)
    
    # Extract key metrics
    spec_stats = speculative_results
    base_stats = baseline_results
    
    spec_latency = spec_stats['latency_stats']['mean']
    base_latency = base_stats['latency_stats']['mean']
    
    # Calculate speedup
    speedup = base_latency / spec_latency if spec_latency > 0 else 0
    
    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 40)
    
    if speedup > 1.0:
        status = f"BENEFICIAL - {speedup:.2f}x SPEEDUP"
        assessment = "Speculative decoding provides performance improvement"
    elif speedup > 0.8:
        status = f"MARGINAL - {speedup:.2f}x (Close to baseline)"
        assessment = "Speculative decoding performs similarly to baseline"
    else:
        status = f"SLOWER - {speedup:.2f}x (Needs optimization)"
        assessment = "Speculative decoding is slower than baseline"
    
    report.append(f"Overall Performance: {status}")
    report.append(f"Assessment: {assessment}")
    report.append("")
    
    # Performance Comparison
    report.append("PERFORMANCE COMPARISON")
    report.append("-" * 40)
    report.append(f"Speculative Decoding:")
    report.append(f"  Mean Latency: {spec_latency:.3f}s")
    report.append(f"  Success Rate: {spec_stats['success_rate']:.1%}")
    report.append(f"  Token Acceptance: {spec_stats['acceptance_rate_stats']['mean']:.1%}")
    report.append("")
    
    report.append(f"Cloud-Only Baseline:")
    report.append(f"  Mean Latency: {base_latency:.3f}s")
    report.append(f"  Success Rate: {base_stats['success_rate']:.1%}")
    report.append("")
    
    report.append(f"SPEEDUP ANALYSIS:")
    report.append(f"  Speedup Ratio: {speedup:.2f}x")
    if speedup > 1.0:
        improvement = (speedup - 1.0) * 100
        report.append(f"  Performance Improvement: +{improvement:.1f}%")
    else:
        degradation = (1.0 - speedup) * 100
        report.append(f"  Performance Degradation: -{degradation:.1f}%")
    report.append("")
    
    # Detailed Metrics
    report.append("DETAILED METRICS")
    report.append("-" * 40)
    
    # Speculative results
    report.append("Speculative Decoding Results:")
    spec_lat = spec_stats['latency_stats']
    report.append(f"  Total Latency: {spec_lat['mean']:.3f}s (±{spec_lat['std_dev']:.3f}s)")
    report.append(f"  Range: {spec_lat['min']:.3f}s - {spec_lat['max']:.3f}s")
    report.append(f"  Tokens Generated: {spec_stats['tokens_generated_stats']['mean']:.1f} avg")
    
    # Add network breakdown if available
    if 'network_latencies' in spec_stats:
        net_lat = spec_stats.get('network_latency_stats', {})
        edge_lat = spec_stats.get('edge_inference_stats', {})
        if net_lat and edge_lat:
            report.append(f"  Network Latency: {net_lat.get('mean', 0):.3f}s")
            report.append(f"  Edge Inference: {edge_lat.get('mean', 0):.3f}s")
    report.append("")
    
    # Baseline results
    report.append("Cloud-Only Baseline Results:")
    base_lat = base_stats['latency_stats']
    report.append(f"  Total Latency: {base_lat['mean']:.3f}s (±{base_lat['std_dev']:.3f}s)")
    report.append(f"  Range: {base_lat['min']:.3f}s - {base_lat['max']:.3f}s")
    report.append(f"  Tokens Generated: {base_stats['tokens_generated_stats']['mean']:.1f} avg")
    
    # Add network breakdown for baseline
    if 'inference_time_stats' in base_stats and 'network_overhead_stats' in base_stats:
        inf_stats = base_stats['inference_time_stats']
        net_stats = base_stats['network_overhead_stats']
        report.append(f"  Pure Inference Time: {inf_stats['mean']:.3f}s")
        report.append(f"  Network Overhead: {net_stats['mean']:.3f}s")
    report.append("")
    
    # Sample outputs comparison
    report.append("SAMPLE OUTPUTS COMPARISON")
    report.append("-" * 40)
    
    spec_samples = spec_stats.get('sample_results', [])
    base_samples = base_stats.get('sample_results', [])
    
    for i in range(min(len(spec_samples), len(base_samples))):
        spec_sample = spec_samples[i]
        base_sample = base_samples[i]
        
        report.append(f"Test {i+1}: {spec_sample['prompt']}")
        report.append(f"  Speculative ({spec_sample['latency']:.2f}s): {spec_sample['generated_text']}")
        
        # Show baseline with network breakdown if available
        baseline_info = f"{base_sample.get('latency', 0):.2f}s"
        if 'inference_time' in base_sample and 'network_overhead' in base_sample:
            baseline_info += f" (inference: {base_sample['inference_time']:.2f}s, network: {base_sample['network_overhead']:.2f}s)"
        
        report.append(f"  Baseline ({baseline_info}): {base_sample['generated_text']}")
        report.append("")
    
    # Conclusions and Recommendations
    report.append("CONCLUSIONS & RECOMMENDATIONS")
    report.append("-" * 40)
    
    if speedup > 1.5:
        report.append("EXCELLENT: Speculative decoding is highly beneficial")
        report.append("   -> Continue using speculative decoding")
        report.append("   -> Consider increasing draft token count for even better performance")
    elif speedup > 1.2:
        report.append("GOOD: Speculative decoding provides meaningful speedup")
        report.append("   -> Continue using speculative decoding")
        report.append("   -> Monitor acceptance rates to ensure continued effectiveness")
    elif speedup > 1.0:
        report.append("MARGINAL: Speculative decoding provides small improvement")
        report.append("   -> Consider optimizing draft model alignment")
        report.append("   -> Evaluate network latency impact")
    else:
        report.append("NOT BENEFICIAL: Speculative decoding is slower than baseline")
        report.append("   -> Check model alignment between draft and target")
        report.append("   -> Optimize network communication")
        report.append("   -> Consider different model combinations")
    
    acceptance_rate = spec_stats['acceptance_rate_stats']['mean']
    if acceptance_rate < 0.3:
        report.append("   -> Low acceptance rate indicates poor draft-target model alignment")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

if __name__ == "__main__":
    # Run the performance test with default parameters
    asyncio.run(RunPerformanceTest())
