echo "installing vllm"
pip install vllm

echo "starting vllm server - gpt-oss-20b"
vllm serve openai/gpt-oss-20b --gpu-memory-utilization 0.92 --enable-prefix-caching --enforce-eager