# Installation Guide

This guide will help you install and set up CCLR-PREDAP on your system.

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 8 GB RAM (16 GB recommended for large datasets)
- **Storage**: At least 2 GB free space

## Installation Methods

### Method 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-organization/CCLR_PREDAP.git
cd CCLR_PREDAP

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Using conda

```bash
# Clone the repository
git clone https://github.com/your-organization/CCLR_PREDAP.git
cd CCLR_PREDAP

# Create conda environment
conda create -n cclr-predap python=3.9
conda activate cclr-predap

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

The main dependencies include:

```text
pandas>=1.3
numpy>=1.21
scipy>=1.7
matplotlib>=3.4
seaborn>=0.11
scikit-learn>=1.0
statsmodels>=0.13
torch>=1.11
darts>=0.24
```

### Optional Dependencies

For enhanced functionality:

```bash
# For Jupyter notebook support
pip install jupyter ipykernel

# For advanced visualization
pip install plotly

# For GPU acceleration (if CUDA available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Verification

Verify your installation by running:

```python
# Test basic imports
from src.lmlr import models_training
from src.gcausal import granger_causation_matrix
from src.dl import create_model_lstm

print("✅ CCLR-PREDAP installation successful!")
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'src'

**Solution**: Make sure you're running Python from the CCLR_PREDAP root directory, or add it to your Python path:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
```

#### CUDA/GPU Issues

**Solution**: For GPU support, ensure you have:

1. Compatible NVIDIA GPU
2. CUDA toolkit installed
3. Correct PyTorch version for your CUDA version

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### Memory Issues

**Solution**: For large datasets:

1. Increase system RAM or use cloud computing
2. Process data in chunks
3. Reduce the number of features in initial analysis

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/your-organization/CCLR_PREDAP/issues)
2. Create a new issue with:
   - Your operating system
   - Python version
   - Complete error message
   - Steps to reproduce

## Development Installation

For contributors:

```bash
git clone https://github.com/your-organization/CCLR_PREDAP.git
cd CCLR_PREDAP

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

## Docker Installation

For containerized deployment:

```bash
# Pull the Docker image
docker pull your-organization/cclr-predap:latest

# Run container
docker run -it -v /path/to/your/data:/data your-organization/cclr-predap:latest
```

## Next Steps

After installation:

1. Check out the [Quick Start Guide](../getting-started/quickstart.md)
2. Explore the [Examples](../examples/basic-usage.md)
3. Read the [User Guide](../user-guide/overview.md) for detailed methodology

---

**Note**: Always use virtual environments to avoid conflicts with other Python packages.