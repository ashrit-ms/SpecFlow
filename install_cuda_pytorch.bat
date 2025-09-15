# PyTorch CUDA Installation Script
# Run this if CUDA is not available in PyTorch

# For CUDA 12.1 (most common)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (if you have older drivers)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# After installation, restart your Python environment and test:
# python -c "import torch; print(torch.cuda.is_available())"
