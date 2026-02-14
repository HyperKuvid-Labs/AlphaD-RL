# AlphaD-RL
Multi-Teacher Monte Carlo Tree Search (MT-MCTS) for code generation, where 3+ diverse teacher models (DeepSeek-Coder, CodeLlama, Qwen2.5-Coder) propose token paths that form the MCTS search tree. The policy network learns to navigate these trees using proper UCB (Q-value + exploration term) with execution rewards from unit tests as the sole signal.
