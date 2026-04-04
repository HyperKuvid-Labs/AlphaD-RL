echo "installing vllm"
pip install vllm

echo "starting vllm server - codestral"
vllm serve mistralai/Codestral-22B-v0.1 --dtype bfloat16 --gpu-memory-utilization 0.92 --enable-prefix-caching --enforce-eager