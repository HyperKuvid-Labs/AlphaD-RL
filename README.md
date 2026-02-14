# AlphaD-RL: Multi-Teacher Monte Carlo Tree Search with Dynamic Weighting and Policy Distillation for Code Generation

Multi-Teacher Monte Carlo Tree Search (MT-MCTS) for code generation, where 3+ diverse teacher models (DeepSeek-Coder, CodeLlama, Qwen2.5-Coder) propose token paths that form the MCTS search tree. The policy network learns to navigate these trees using proper UCB (Q-value + exploration term) with execution rewards from unit tests as the sole signal.

**Student model**: Qwen/Qwen2.5-Coder-7B-Instruct (pass@1: 22/164 on HumanEval)

**Teacher model**: Qwen/Qwen2.5-Coder-7B-Instruct, deepseek-ai/deepseek-coder-6.7b-instruct, meta-llama/CodeLlama-7b-Instruct-hf

There are two phases in here:
- training the model to determine, which level is enough for the token level, and generating the whole function from there
- mcts planned teacher populated tree search with boiling down to path of selected and rejected, and dpo optimization of local model, adn then after iteration update the main target model.

