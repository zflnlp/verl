#!/bin/bash
# Fix flash_attn on the server (CUDA 13.1, Python 3.12)

# Remove the broken flash_attn
pip uninstall flash_attn -y
rm -rf /usr/local/lib/python3.12/dist-packages/flash_attn*
rm -rf /usr/local/lib/python3.12/dist-packages/flash_attn-*

# Reinstall from source to match current PyTorch
MAX_JOBS=4 pip install flash_attn --no-build-isolation

# Verify
python -c "import flash_attn; print('flash_attn OK')"
