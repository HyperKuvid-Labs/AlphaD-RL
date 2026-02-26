echo "installing vllm"
pip install vllm

echo "starting vllm server - qwen2.5-coder-14b-instruct"
vllm serve Qwen/Qwen2.5-Coder-14B-Instruct --max-model-len 4096 --tensor-parallel-size 1