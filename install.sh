# echo "installing uv"
# sudo curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt install python-is-python3
sudo apt install python3-pip

echo "solving dependency error of this: CRITICAL WARNING: PyTorch 2.9.1 & CuDNN Compatibility Issue Detected"
pip install nvidia-cudnn-cu12==9.16.0.29

# echo "syncing uv"
# uv sync
echo "installing torch and transformers and tensorboard"
pip install vllm
pip install torch transformers datasets trl tensorboard bitsandbytes

