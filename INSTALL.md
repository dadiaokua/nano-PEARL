# Installation Guide

## Quick Start

### Option 1: Install from source (Recommended)

```bash
# 1. First install torch (required for flash-attn)
pip install torch>=2.4.0

# 2. Then install the rest
pip install -r requirements.txt

# Or install the package itself
pip install -e .
```

### Option 2: Step-by-step installation

If you encounter issues with flash-attn, install dependencies step by step:

```bash
# Step 1: Install torch first (CRITICAL for flash-attn)
pip install torch>=2.4.0 triton>=3.0.0

# Step 2: Install other dependencies
pip install transformers>=4.51.0 xxhash rich>=14.1.0 numpy>=1.24.0 tqdm>=4.65.0

# Step 3: Install flash-attn (requires torch to be already installed)
pip install flash-attn

# Step 4: Install optional GPU power monitoring (Linux only)
pip install nvidia-ml-py3>=11.0.0
```

### Option 3: Using uv (Faster)

```bash
# uv handles dependencies better
uv pip install -e .
```

## Troubleshooting

### Flash Attention Installation Issues

If `flash-attn` installation fails:

1. **Make sure torch is installed first:**
   ```bash
   pip install torch>=2.4.0
   ```

2. **Try installing without build isolation:**
   ```bash
   pip install flash-attn --no-build-isolation
   ```

3. **Or install from pre-built wheel:**
   ```bash
   # Check available wheels at: https://github.com/Dao-AILab/flash-attention/releases
   pip install flash-attn --no-build-isolation --no-deps
   ```

4. **Build from source (if needed):**
   ```bash
   pip install ninja
   MAX_JOBS=4 pip install flash-attn --no-build-isolation
   ```

### GPU Power Monitoring

- `nvidia-ml-py3` is only available on Linux
- On Windows/Mac, GPU power monitoring will be automatically disabled
- The code will work fine without it, just without power monitoring features

## Requirements

- Python >= 3.12
- CUDA-capable GPU (for flash-attn)
- Linux (recommended) or macOS/Windows (with limitations)

