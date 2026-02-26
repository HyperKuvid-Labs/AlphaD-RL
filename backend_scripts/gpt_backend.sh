echo "installing vllm"
pip install vllm

echo "starting vllm server - gpt-oss-20b"
vllm serve openai/gpt-oss-20b --max-model-len 4096 --tensor-parallel-size 1