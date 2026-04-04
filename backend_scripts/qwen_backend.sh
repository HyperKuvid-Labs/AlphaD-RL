echo "installing vllm"
pip install vllm

echo "starting vllm server - qwen2.5-coder-14b-instruct"
vllm serve Qwen/Qwen2.5-Coder-14B-Instruct --dtype bfloat16 --gpu-memory-utilization 0.92 --enable-prefix-caching --enforce-eager