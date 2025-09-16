# SpecECD: Speculative Edge-Cloud Decoding

Implementation of speculative decoding with corrected probabilistic verification based on "Fast Inference from Transformers via Speculative Decoding".

## Overview

This project implements a distributed inference system where:
- **Edge Device**: Runs a small draft model (Llama-3.2-1B-Instruct) to generate speculative tokens with probabilities
  - **CPU Support**: PyTorch-based inference for broad compatibility
  - **GPU Support**: CUDA acceleration for faster edge inference
  - **NPU Support**: OpenVINO NPU acceleration for Intel Arc GPUs and dedicated NPU hardware
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
- **Acceleration Options**:
  - **CPU**: PyTorch inference (default, compatible with all systems)
  - **GPU**: CUDA acceleration (NVIDIA GPUs, faster inference)
  - **NPU**: OpenVINO NPU acceleration (Intel Arc GPUs, dedicated NPU hardware)

### Cloud (Target Model)  
- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Parameters**: 8B
- **Features**: Probabilistic verification using single forward pass

## Setup Instructions

### Prerequisites
```bash
# Install Python dependencies
pip install -r requirements.txt

# For GPU support, install appropriate PyTorch version with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For OpenVINO NPU acceleration (optional)
pip install openvino openvino-dev optimum[openvino]
```

### GPU Setup (Optional)

CUDA GPU provides significant acceleration for edge inference:

#### 1. Hardware Requirements
- **NVIDIA GPU**: GeForce RTX, Quadro, or Tesla series with CUDA support
- **CUDA Drivers**: Latest NVIDIA drivers (minimum 11.8)
- **Memory**: At least 4GB VRAM for 1B model

#### 2. Check GPU Availability
```bash
# List available edge devices (includes GPU info)
python run_edge.py --list-devices

# Test GPU functionality
python test_gpu.py
```

#### 3. Enable GPU in Configuration
Edit `config.toml`:
```toml
[devices]
edge_device = "gpu"  # Force GPU usage

# Or enable GPU with settings
[devices.gpu]
enabled = true       # Try GPU, fallback to CPU if unavailable
device_id = 0        # Use first GPU
```

### OpenVINO NPU Setup (Optional)

OpenVINO NPU provides hardware acceleration for Intel Arc GPUs and dedicated NPU chips:

#### 1. Check NPU Availability
```bash
# List available edge devices
python run_edge.py --list-devices

# Test NPU functionality
python test_npu.py
```

#### 2. Enable NPU in Configuration
Edit `config.toml`:
```toml
[devices]
edge_device = "npu"  # Force NPU usage

# Or enable NPU with fallback
[devices.npu]
enabled = true       # Try NPU, fallback to CPU if unavailable
```

#### 3. Hardware Requirements
- **Intel Arc GPUs**: Intel Arc A-series with NPU support
- **Intel CPUs**: 12th gen Core or newer with built-in NPU
- **Dedicated NPU**: Intel Movidius or similar accelerators
- **Drivers**: Latest Intel graphics drivers and OpenVINO runtime

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
# Connect to localhost (testing) - uses device from config
python run_edge.py

# Connect with specific device
python run_edge.py --device cpu   # Force CPU inference
python run_edge.py --device gpu   # Force GPU acceleration
python run_edge.py --device npu   # Force NPU acceleration

# Connect to remote server with GPU
python run_edge.py --host 192.168.1.100 --device gpu

# Check available devices
python run_edge.py --list-devices
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
├── config.toml               # Configuration file with device options
├── run_cloud.py              # Cloud server entry point
├── run_edge.py               # Edge client entry point  
├── run_tests.py              # Performance test entry point
├── test_npu.py               # NPU functionality tests
├── test_gpu.py               # GPU functionality tests
├── common/
│   ├── config.py             # Configuration loader with GPU/NPU support
│   └── protocol.py           # Shared data structures and utilities
├── edge/
│   ├── draft_model.py        # Edge draft model with CPU/GPU/NPU support
│   ├── openvino_model.py     # OpenVINO NPU model wrapper
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
- **Edge inference time**: Draft model processing (CPU, GPU, or NPU)
- **Component breakdown**: Detailed timing analysis

### Expected Results
| Configuration | Speedup | Acceptance Rate | Notes |
|---------------|---------|-----------------|-------|
| **Cloud Only** | 1.0x | - | Baseline performance |
| **Edge CPU + Cloud** | 2-3x | 60-80% | Standard speculative decoding |
| **Edge GPU + Cloud** | 3-4x | 60-80% | GPU acceleration benefit |
| **Edge NPU + Cloud** | 3-5x | 60-80% | NPU acceleration benefit |

### Hardware Acceleration Benefits
- **CPU**: Broad compatibility, no special hardware required
- **GPU**: High throughput, significant speedup over CPU (2-5x faster)
- **NPU**: Lower power consumption, optimized for AI inference

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
4. **Multi-Device Support**: CPU (PyTorch), GPU (CUDA), and NPU (OpenVINO) acceleration
5. **Flexible Configuration**: Easy device switching via config or command line
6. **Automatic Fallback**: GPU/NPU fall back to CPU if hardware unavailable
7. **Flexible Testing**: Configurable number of prompts and iterations
8. **Real-time Communication**: WebSocket-based low-latency protocol
9. **Performance Monitoring**: Comprehensive metrics and assessment

## Troubleshooting

### Common Issues
1. **Connection Failed**: Check firewall settings and IP addresses
2. **Model Loading Error**: Ensure sufficient RAM/VRAM available
3. **Low Acceptance Rate**: May indicate model mismatch or algorithm issues
4. **CUDA Issues**: Verify PyTorch CUDA installation
5. **GPU Not Available**: Run `python test_gpu.py` to check GPU status
6. **NPU Not Available**: Run `python test_npu.py` to check NPU status

### GPU Issues
1. **GPU Not Detected**:
   - Check hardware compatibility (NVIDIA GPU with CUDA support)
   - Update NVIDIA drivers
   - Verify CUDA installation: `nvidia-smi`
   - Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

2. **CUDA Out of Memory**:
   - Close other GPU applications to free memory
   - Try CPU fallback: `--device cpu`
   - Monitor GPU memory usage: `nvidia-smi`

3. **Performance Lower Than Expected**:
   - Check if GPU is actually being used in model info output
   - Monitor GPU utilization: `nvidia-smi -l 1`
   - Compare with CPU baseline: `python run_edge.py --device cpu`

### OpenVINO NPU Issues
1. **NPU Not Detected**: 
   - Check hardware compatibility (Intel Arc GPU or NPU-enabled CPU)
   - Update Intel graphics drivers
   - Verify OpenVINO installation: `python -c "import openvino; print('OpenVINO OK')"`

2. **Model Conversion Failed**:
   - Ensure sufficient disk space for IR model files
   - Check internet connection for model download
   - Try CPU fallback: `--device cpu`

3. **Performance Lower Than Expected**:
   - Check if NPU is actually being used in model info output
   - Verify no other processes are using NPU
   - Compare with CPU baseline: `python run_edge.py --device cpu`

### Performance Tips
1. Use GPU acceleration when available on cloud server
2. Monitor acceptance rates - should be >30% for effective speedup
3. **NPU**: Use for edge inference, especially on battery-powered devices
4. Adjust temperature and sampling parameters if needed
5. Ensure models are properly loaded before testing

## Research Applications

This implementation demonstrates:
- Correct speculative decoding algorithm implementation
- Edge-cloud distributed inference patterns
- Multi-device acceleration (CPU, NPU) for edge AI
- Probabilistic token verification methods
- Performance analysis and benchmarking across hardware platforms
- Model combination effectiveness evaluation
- OpenVINO integration for AI acceleration
