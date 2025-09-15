# SpecECD: Speculative Edge-Cloud Decoding

Implementation of speculative decoding with corrected probabilistic verification based on "Fast Inference from Transformers via Speculative Decoding".

## Overview

This project implements a distributed inference system where:
- **Edge Device**: Runs a small draft model (Llama-3.2-1B-Instruct) to generate speculative tokens with probabilities
- **Cloud Server**: Runs a larger target model (Llama-3.1-8B-Instruct) for probabilistic verification and completion
- **Communication**: WebSocket-based protocol for real-time coordination

## Architecture

```
┌─────────────────┐    WebSocket     ┌─────────────────┐
│   Edge Client   │◄────────────────►│  Cloud Server   │
│  (Draft Model)  │    Verification  │ (Target Model)  │
│     1B params   │     Requests     │    8B params    │
└─────────────────┘                  └─────────────────┘
```

## Model Selection

### Edge (Draft Model)
- **Model**: meta-llama/Llama-3.2-1B-Instruct
- **Parameters**: 1.2B
- **Features**: Probability tracking for each generated token

### Cloud (Target Model)  
- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Parameters**: 8B
- **Features**: Probabilistic verification using single forward pass

## Setup Instructions

### Prerequisites
```bash
# Install Python dependencies
pip install -r requirements.txt

# For GPU support, install appropriate PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Network Configuration
1. Ensure both machines are on the same local network
2. Configure firewall to allow traffic on port 8765
3. Update IP addresses in the scripts

## Usage

### 1. Start Cloud Server
```bash
python run_cloud.py
```

### 2. Run Edge Client
```bash
# Connect to localhost (testing)
python run_edge.py

# Connect to remote server
python run_edge.py <server_ip_address>
```

### 3. Run Performance Tests
```bash
# Test with default 5 prompts, 2 iterations each
python run_tests.py

# Test with custom number of prompts
python run_tests.py --num-prompts 3 --iterations 1

# Test with remote server
python run_tests.py --host 192.168.1.100 --port 8765
```

## Test Options

### Command Line Arguments
- `--num-prompts N`: Number of test prompts to use (1-10, default: 5)
- `--iterations N`: Number of iterations per prompt (default: 2)
- `--host HOST`: Cloud server host (default: localhost)
- `--port PORT`: Cloud server port (default: 8765)

### Test Prompts
The system tests diverse prompts including:
- AI and technology topics
- Scientific discoveries
- Programming tasks
- General knowledge topics

## File Structure

```
SpecECD/
├── requirements.txt           # Python dependencies
├── run_cloud.py              # Cloud server entry point
├── run_edge.py               # Edge client entry point  
├── run_tests.py              # Performance test entry point
├── common/
│   └── protocol.py           # Shared data structures and utilities
├── edge/
│   ├── draft_model.py        # Edge draft model with probability tracking
│   └── client.py             # Edge client with WebSocket communication
├── cloud/
│   ├── target_model.py       # Cloud target model with probabilistic verification
│   └── server.py             # Cloud server with WebSocket handling
└── tests/
    └── performance_test.py   # Performance evaluation suite
```

## Corrected Algorithm

This implementation includes the **corrected speculative decoding algorithm**:

### Draft Model (Edge)
- Generates tokens using autoregressive sampling
- **Tracks probability for each generated token**
- Sends both tokens and probabilities to cloud

### Target Model (Cloud)
- **Single forward pass** on prompt + draft tokens
- For each draft token: `p_accept = min(1.0, p_target / p_draft)`
- **Probabilistic acceptance** using random sampling
- Generates new token only when rejecting

### Key Improvements
- ✅ **Probabilistic verification** (not string matching)
- ✅ **Single forward pass** (not double inference)  
- ✅ **Proper probability tracking** at each step
- ✅ **Corrected acceptance sampling** algorithm

## Performance Metrics

The system measures and reports:

- **End-to-end latency**: Total generation time
- **Token acceptance rate**: Critical metric for algorithm effectiveness
- **Network latency**: Communication overhead
- **Edge inference time**: Draft model processing
- **Component breakdown**: Detailed timing analysis

### Expected Results (Corrected Algorithm)
- **2-3x speedup** vs cloud-only inference
- **60-80% token acceptance rate** (vs 10-30% with broken algorithm)
- **Single inference pass** reducing computational overhead
- **High quality text generation** maintained

## Performance Assessment

The test report includes automatic assessment:
- **EXCELLENT**: >50% acceptance rate (algorithm working correctly)
- **GOOD**: >30% acceptance rate (significant improvement)  
- **MODERATE**: >10% acceptance rate (some improvement)
- **POOR**: <10% acceptance rate (needs further tuning)

## Key Features

1. **Corrected Speculative Decoding**: Implements proper probabilistic verification
2. **Probability Tracking**: Draft model tracks and sends token probabilities
3. **Single Forward Pass**: Target model processes all draft tokens at once
4. **Flexible Testing**: Configurable number of prompts and iterations
5. **Real-time Communication**: WebSocket-based low-latency protocol
6. **Performance Monitoring**: Comprehensive metrics and assessment

## Troubleshooting

### Common Issues
1. **Connection Failed**: Check firewall settings and IP addresses
2. **Model Loading Error**: Ensure sufficient RAM/VRAM available
3. **Low Acceptance Rate**: May indicate model mismatch or algorithm issues
4. **CUDA Issues**: Verify PyTorch CUDA installation

### Performance Tips
1. Use GPU acceleration when available
2. Monitor acceptance rates - should be >30% for effective speedup
3. Adjust temperature and sampling parameters if needed
4. Ensure models are properly loaded before testing

## Research Applications

This implementation demonstrates:
- Correct speculative decoding algorithm implementation
- Edge-cloud distributed inference patterns
- Probabilistic token verification methods
- Performance analysis and benchmarking
- Model combination effectiveness evaluation
