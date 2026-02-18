echo "running pip install sglang"
pip install sglang

echo "solving dependency error of this: CRITICAL WARNING: PyTorch 2.9.1 & CuDNN Compatibility Issue Detected"
pip install nvidia-cudnn-cu12==9.16.0.29

