echo "installing vllm"
pip install vllm

echo "starting vllm server - codestral"
vllm serve mistralai/Codestral-22B-v0.1 --max-model-len 4096 --tensor-parallel-size 1 