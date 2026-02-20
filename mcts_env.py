import os
import math
import subprocess
import re
from .utils import generate_best_solution, expand_leaf, get_process_reward, get_all_leaf_nodes, get_top_3_leaves, get_drift_reward, get_cr, get_pr

class Node:
    # basic mcts node structure
    def __init__(self, token_id, generated_text, parent):
        self.token_id = token_id
        self.generated_text = generated_text
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.value = 0.0

    def add_child(self, child_node):
        self.children.append(child_node)

    def get_ucb_score(self):
        # using ucb1 instead of puct cause logs might be trash and we need hard exploration
        c = 1.414
        if self.visit_count > 0:
            exploitation = self.value / self.visit_count
            exploration = c * (math.sqrt(math.log(self.parent.visit_count) / self.visit_count))
        else:
            return float('inf')
        return exploitation + exploration

class MCTSEnvironment:
    def __init__(self, tm1, tm2, tm3, hf_tm1, hf_tm2, hf_tm3, tok1, tok2, tok3, params1, params2, params3):
        # load all the heavy teacher models and tokenizers once when the worker boots up
        # no student models here, verl handles the student on a separate ray worker
        self.tm1 = tm1
        self.tm2 = tm2
        self.tm3 = tm3
        self.hf_tm1 = hf_tm1
        self.hf_tm2 = hf_tm2
        self.hf_tm3 = hf_tm3
        self.tok1 = tok1
        self.tok2 = tok2
        self.tok3 = tok3
        self.params1 = params1
        self.params2 = params2
        self.params3 = params3

        # internal state trackers
        self.current_prompt = ""
        self.current_test = ""
        self.current_entrypoint = ""
        self.root_node = None
        self.golden_solution = ""
        self.step_count = 0
        self.max_steps = 10 # hard cap so it doesnt spin forever

    def reset(self, prompt, test, entrypoint):
        self.current_prompt = prompt
        self.current_test = test
        self.current_entrypoint = entrypoint
        self.step_count = 0

        # init the root
        self.root_node = Node(token_id=None, generated_text="", parent=None)

        # grab golden solution for process rewards later
        self.golden_solution = generate_best_solution(
            prompt, self.tm1, self.tm2, self.tm3,
            self.params1, self.params2, self.params3
        )

        # build the very first state observation for the level guesser
        # at root, length is 0, no agreement yet, etc
        initial_state = f"Length:0, Agree:False, Value:0.00, Nodes:0. Stop? (Yes/No):"
        return initial_state

    def step(self, action_text: str):
        action = action_text.strip().lower()
        self.step_count += 1

        # if the actor says 'yes', or we hit the max depth, we stop the tree and evaluate
        if "yes" in action or self.step_count >= self.max_steps:
            return self._terminate_and_evaluate()

        # if the actor says 'no', fuck it we keep expanding
        # 1. select leaf
        leaf = self._select_leaf_node(self.root_node)

        # 2. expand it using the hf logprob models
        teachers_agreement = expand_leaf(
            leaf, self.current_prompt,
            self.hf_tm1, self.hf_tm2, self.hf_tm3,
            self.tok1, self.tok2, self.tok3
        )

        # 3. get process reward and backprop up the tree
        reward = get_process_reward(
            self.current_prompt, self.golden_solution, leaf.generated_text,
            self.tm1, self.tm2, self.tm3, self.tok1, self.tok2, self.tok3
        )
        self._backpropagate(leaf, reward)

        # 4. construct the new state for the actor to look at next
        seq_length = len(leaf.generated_text)
        if leaf.parent is not None:
            siblings = leaf.parent.children
            node_count = len(siblings)
            avg_value = sum(s.value for s in siblings) / node_count if node_count > 0 else 0.0
        else:
            node_count = len(leaf.children)
            avg_value = leaf.value

        next_state = f"Length:{seq_length}, Agree:{teachers_agreement}, Value:{avg_value:.2f}, Nodes:{node_count}. Stop? (Yes/No):"

        # return state, 0 reward for intermediate step, done=False
        return next_state, 0.0, False

    def _terminate_and_evaluate(self):
        """
        actor chose to stop. run the unit tests and calculate the final grpo reward.
        """
        all_leaves = get_all_leaf_nodes(self.root_node)
        num_leaves = len(all_leaves)
        top_3_nodes = get_top_3_leaves(all_leaves)

        # ask teachers to complete the code from the top 3 nodes
        continuation_prompts = []
        for node in top_3_nodes:
            formatted_prompt = f"### Context\n{self.current_prompt}\n### Partial Implementation\n{node.generated_text}\n### Instructions\n1. Complete the code starting exactly from where the partial implementation ends. Do NOT repeat the existing code.\n2. Integrate the Entrypoint: Ensure the logic flows into the `{self.current_entrypoint}` function.\n3. Main Function & Testing: Append a `if __name__ == '__main__':` block.\n4. Validation: Use the following test cases: {self.current_test}.\n5. Output Format: The script must conclude by printing the exact string: 'Passed X out of Y test cases'.\n### Completion:"
            continuation_prompts.append(formatted_prompt)

        outputs1 = self.tm1.generate(continuation_prompts, self.params1)
        outputs2 = self.tm2.generate(continuation_prompts, self.params2)
        outputs3 = self.tm3.generate(continuation_prompts, self.params3)

        lengths = []
        test_passed_reward = 1.0
        os.makedirs("temp_scripts_mcts", exist_ok=True)

        for i in range(len(top_3_nodes)):
            script1 = outputs1[i]['text']
            script2 = outputs2[i]['text']
            script3 = outputs3[i]['text']

            count_passed = 0
            file_paths = [
                f"temp_scripts_mcts/script1_node{i}.py",
                f"temp_scripts_mcts/script2_node{i}.py",
                f"temp_scripts_mcts/script3_node{i}.py"
            ]

            with open(file_paths[0], "w") as f: f.write(script1)
            with open(file_paths[1], "w") as f: f.write(script2)
            with open(file_paths[2], "w") as f: f.write(script3)

            outputs = []
            for path in file_paths:
                try:
                    res = subprocess.run(["python", path], capture_output=True, text=True, timeout=5)
                    outputs.append(res.stdout)
                except subprocess.TimeoutExpired:
                    outputs.append("")

            # clean up temp files immediately
            for path in file_paths:
                if os.path.exists(path): os.remove(path)

            for output in outputs:
                match = re.search(r"Passed\s+(\d+)\s+out\s+of\s+(\d+)\s+test\s+cases", output)
                if match:
                    passed, total = int(match.group(1)), int(match.group(2))
                    if passed == total and total > 0:
                        count_passed += 1

            if count_passed < 1 and test_passed_reward != -1.0:
                test_passed_reward = -1.0

            lengths.append((len(script1), len(script2), len(script3)))

        # calculate the drift reward and prune reward based on your formulas
        min_len = min(min(lens) for lens in lengths)
        avg_len = sum(sum(lens) for lens in lengths) / (3 * len(lengths))
        max_len = max(max(lens) for lens in lengths)

        cr = get_drift_reward(min_len, avg_len, max_len, lengths)
        final_cr = get_cr(cr, test_passed_reward)
        pr = get_pr(num_leaves)

        # total final reward for the level guesser trajectory
        total_reward = final_cr + pr

        # return dummy final state, total reward, done=true
        return "Episode Finished", total_reward, True

    # --- internal helpers wrapped from your script ---
    def _select_leaf_node(self, root_node):
        current_node = root_node
        while len(current_node.children) != 0:
            next_node = None
            max_ucb = -float('inf')
            for child in current_node.children:
                ucb_score = child.get_ucb_score()
                if ucb_score > max_ucb:
                    max_ucb = ucb_score
                    next_node = child
            current_node = next_node
        return current_node

    def _backpropagate(self, node, reward):
        node.visit_count += 1
        node.value += reward
        if node.parent is not None:
            self._backpropagate(node.parent, reward)