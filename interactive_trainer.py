"""
interactive_trainer.py

Human-in-the-loop GRPO trainer.

Instead of calling remote teacher-model servers, every oracle call is routed
back to *you* at the terminal:

  • Best solution  → taken directly from the dataset (best_solution field)
  • Token expansion → you type next tokens (comma-separated) with optional
                      log-probability scores
  • Process reward  → you rate the partial solution on [-1, 1]

The student (Qwen/Qwen3-4B), GRPO loss, optimiser and TensorBoard logging all
work exactly as in level_guesser_trainer.py.

Run:
    python interactive_trainer.py
"""

from __future__ import annotations

import html
import json
import math
import os
import random
import re
import subprocess
import sys
import textwrap
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import urllib.error
import urllib.request

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset

# ── rich imports ──────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
    from rich.rule import Rule
    from rich.prompt import Prompt, Confirm
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn,
    )
    from rich.columns import Columns
    from rich.align import Align
    from rich.live import Live
    from rich.markup import escape
    from rich import box
except ImportError:
    print("rich is not installed.  Run:  pip install rich")
    sys.exit(1)

console = Console()

# ──────────────────────────────────────────────────────────────────────────────
# Teacher model servers  (one entry per teacher, each at its own IP/port)
# Mirrors the layout from quant_init.sh / test_get_30_tokens_vllm.py
# Override individual base_url values or add/remove entries as needed.
# ──────────────────────────────────────────────────────────────────────────────

TEACHER_MODELS: List[dict] = [
    {
        "display_name": "GPT-oss-20B",
        "name":         "openai/gpt-oss-20b",
        "base_url":     os.environ.get("VLLM_URL_0", "http://100.73.177.66:8002"),
    },
    {
        "display_name": "Qwen2.5-Coder-14B",
        "name":         "Pradheep1647/adrl-qwen2.5-coder-4bit",
        "base_url":     os.environ.get("VLLM_URL_1", "http://100.106.99.109:8000"),
    },
    {
        "display_name": "Codestral-22B",
        "name":         "Pradheep1647/adrl-codestral-4bit",
        "base_url":     os.environ.get("VLLM_URL_2", "http://100.127.121.101:8001"),
    },
]


def _get_next_token_logprobs_vllm(
    model_name: str,
    base_url: str,
    prompt: str,
    top_n: int = 20,
) -> List[Tuple[str, float]]:
    """
    Query a vLLM ``/v1/completions`` server for the log-prob distribution
    over the next token (mirrors ``get_next_token_logprobs_vllm`` in
    test_get_30_tokens_vllm.py).

    Returns a list of (token_str, logprob) sorted descending by logprob.
    Raises on HTTP / parse errors so callers can catch and fall back.
    """
    payload = {
        "model":       model_name,
        "prompt":      prompt,
        "max_tokens":  1,
        "temperature": 0.0,
        "logprobs":    max(top_n, 1),
        "echo":        False,
    }

    if _REQUESTS_AVAILABLE:
        resp = _requests.post(
            f"{base_url}/v1/completions", json=payload, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
    else:
        raw = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{base_url}/v1/completions",
            data=raw,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode())

    choice           = data["choices"][0]
    logprobs_obj     = choice.get("logprobs", {})
    top_logprobs_list = logprobs_obj.get("top_logprobs", [])

    if not top_logprobs_list:
        raise ValueError(f"No logprobs returned from {model_name} at {base_url}")

    token_logprob_map = top_logprobs_list[0]
    distribution: List[Tuple[str, float]] = []
    for token_id_str, info in token_logprob_map.items():
        if isinstance(info, dict):
            lp  = info.get("logprob", float("-inf"))
            tok = info.get("decoded_token", info.get("token", str(token_id_str)))
        else:
            lp  = float(info)
            tok = str(token_id_str)
        distribution.append((tok, lp))

    distribution.sort(key=lambda x: x[1], reverse=True)
    return distribution


def get_30_tokens_vllm(
    prompt: str,
    top_k: int = 5,
) -> Tuple[List[Tuple[str, str, str, float]], bool]:
    """
    Collect ``top_k`` best + ``top_k`` worst next-token candidates from every
    teacher model (mirrors ``get_30_tokens_vllm`` in test_get_30_tokens_vllm.py).

    Returns
    -------
    all_tokens        : list of (display_name, rank_label, token, logprob)
    teachers_agreement: True when every reachable teacher agrees on top-1 token
    """
    all_tokens: List[Tuple[str, str, str, float]] = []
    top1_tokens: List[str] = []

    for model_cfg in TEACHER_MODELS:
        try:
            distribution = _get_next_token_logprobs_vllm(
                model_cfg["name"], model_cfg["base_url"], prompt
            )
        except Exception as exc:
            warn(f"[{model_cfg['display_name']}] vLLM call failed: {exc}")
            continue

        top_tokens    = distribution[:top_k]
        bottom_tokens = distribution[-top_k:][::-1]

        top1_tokens.append(top_tokens[0][0])

        for rank, (tok, lp) in enumerate(top_tokens, start=1):
            all_tokens.append((model_cfg["display_name"], f"top-{rank}",    tok, lp))
        for rank, (tok, lp) in enumerate(bottom_tokens, start=1):
            all_tokens.append((model_cfg["display_name"], f"bottom-{rank}", tok, lp))

    teachers_agreement = len(set(top1_tokens)) == 1 if len(top1_tokens) >= 2 else True
    return all_tokens, teachers_agreement


def _fetch_tokens_vllm(
    partial_code: str,
    n: int,
    teacher_idx: int,
) -> Optional[List[Tuple[str, float]]]:
    """
    Fetch the top-*n* next-token candidates for the teacher at *teacher_idx*
    using the ``/v1/completions`` endpoint at that teacher's dedicated IP/port.

    Returns a list of (token, logprob) pairs on success, or *None* so the
    caller can fall back to human input.
    """
    if teacher_idx >= len(TEACHER_MODELS):
        return None

    model_cfg  = TEACHER_MODELS[teacher_idx]
    model_name = model_cfg["name"]
    base_url   = model_cfg["base_url"]

    try:
        distribution = _get_next_token_logprobs_vllm(
            model_name, base_url, partial_code, top_n=max(n, 20)
        )
        return distribution[:n] if distribution else None
    except Exception as exc:
        warn(
            f"vLLM call failed [{model_cfg['display_name']} @ {base_url}]: "
            f"{exc}  → falling back to human input"
        )
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InteractiveTrainConfig:
    # ── student ──────────────────────────────────────────────────────────────
    student_model_id: str       = "Qwen/Qwen3-4B"
    student_max_new_tokens: int = 5
    student_temperature: float  = 0.9

    # ── GRPO hypers ───────────────────────────────────────────────────────────
    group_size: int    = 2         # keep small — you are the oracle
    clip_eps: float    = 0.2
    kl_coeff: float    = 0.01
    entropy_coeff: float = 0.01
    lr: float          = 2e-5
    warmup_steps: int  = 10
    grad_clip: float   = 1.0
    grad_accum: int    = 2

    # ── training loop ─────────────────────────────────────────────────────────
    max_problems: int  = 20        # interactive → keep the set small
    save_every: int    = 5
    save_dir: str      = "checkpoints_interactive"
    log_dir: str       = "runs/alphaD_rl_interactive"

    # ── MCTS ──────────────────────────────────────────────────────────────────
    max_steps: int     = 8         # hard cap on MCTS depth per rollout
    tokens_per_expand: int = 5     # how many tokens you give per expansion

    # ── teacher names (display only — kept in sync with TEACHER_MODELS) ────────
    teacher_names: List[str] = field(
        default_factory=lambda: [m["display_name"] for m in TEACHER_MODELS]
    )

    # ── human-input logging ───────────────────────────────────────────────────
    log_inputs_path: str = "human_inputs.jsonl"  # set to "" to disable


# ──────────────────────────────────────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────────────────────────────────────

BANNER = r"""
   ___  _       _          ___        ___ _
  / _ \| |_ __ | |__   __ |   \      | _ \ |
 | (_) | | '_ \| '_ \ / _` | |) |   |   / |__
  \__,_|_| .__/|_| |_|\__,_|___/    |_|_\____|
          |_|    Interactive Trainer  v1.0
"""

THEME = {
    "title":    "bold cyan",
    "header":   "bold yellow",
    "ok":       "bold green",
    "warn":     "bold yellow",
    "err":      "bold red",
    "dim":      "dim white",
    "code":     "bright_white on grey15",
    "prompt":   "bold magenta",
    "state":    "bold blue",
    "reward":   "bold green",
}

LEVEL_GUESSER_EXPLANATION = """
[bold cyan]Level Guesser – what we are actually training[/]

This is a small policy we train with reinforcement learning.

Its only job:

Decide whether we should [bold]STOP[/] expanding the current partial code
and evaluate the programs we can complete from here — or whether we should
[bold]KEEP GOING[/] because the partial is still too uncertain / incomplete.

We want it to learn to say [bold green]Yes (stop)[/] when:

• the partial already shows a correct high-level algorithm / structure
• strong models (teachers) mostly agree what should come next
• continuing would most likely not find dramatically better solutions
• the direction looks compatible with the optimal time complexity

And say [bold yellow]No (continue)[/] when:

• the partial is still ambiguous, buggy or clearly missing key parts
• teachers disagree a lot → high uncertainty
• we are still early → more search is likely valuable

In one sentence:

[italic]Level Guesser learns to recognize the moment when further search
is unlikely to be worth the extra tokens — imitating a very strong but lazy expert.[/]
"""


def banner() -> None:
    console.print(Panel(
        Align.center(Text(BANNER, style="bold cyan")),
        border_style="cyan",
        padding=(0, 4),
    ))


def section(title: str) -> None:
    console.print()
    console.print(Rule(f"[{THEME['header']}]{title}[/]", style="yellow"))


def info(msg: str) -> None:
    console.print(f"  [{THEME['ok']}]✓[/]  {msg}")


def warn(msg: str) -> None:
    console.print(f"  [{THEME['warn']}]⚠[/]  {msg}")


def _read_multiline(label: str, end_marker: str = "END") -> str:
    """
    Display a styled prompt then collect lines until the user types `end_marker`
    alone on a line.  Returns the collected text (without the sentinel).
    """
    console.print(Panel(
        f"[{THEME['prompt']}]{label}[/]\n"
        f"[{THEME['dim']}]Paste / type your input.  "
        f"Type [bold]{end_marker}[/] on its own line when done.[/]",
        border_style="magenta",
        title="[bold magenta]Input[/]",
    ))
    lines: List[str] = []
    try:
        while True:
            line = input()
            if line.strip() == end_marker:
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines)


def _read_float(prompt: str, lo: float = -1.0, hi: float = 1.0) -> float:
    """Read a float in [lo, hi], retrying on invalid input."""
    while True:
        raw = Prompt.ask(f"  [{THEME['prompt']}]{prompt}[/]")
        try:
            v = float(raw.strip())
            if lo <= v <= hi:
                return v
            warn(f"Value must be in [{lo}, {hi}].  Got {v}.")
        except ValueError:
            warn(f"Not a valid number: {raw!r}")


def _read_teacher_responses(
    teacher_names: List[str],
    action_label: str,
    hint: str = "",
) -> List[str]:
    """
    Ask the user to paste a response from each teacher model in turn.
    Returns a list of 3 raw strings (one per teacher).
    """
    responses: List[str] = []
    for i, name in enumerate(teacher_names):
        console.print(Panel(
            f"[{THEME['header']}]Teacher {i+1} / {len(teacher_names)}  —  {name}[/]\n"
            + (f"[{THEME['dim']}]{hint}[/]" if hint else ""),
            border_style="yellow",
            title=f"[bold yellow]{action_label}[/]",
        ))
        resp = _read_multiline(f"Paste the response from [bold]{name}[/]:")
        responses.append(resp)
    return responses


def _read_tokens(prompt_text: str, n: int) -> List[Tuple[str, float]]:
    """
    Ask for up to *n* tokens with optional log-prob scores.

    Expected format (one per line):
        token          →  score defaults to 0.0
        token : -0.45  →  explicit log-prob

    Returns [(token_text, log_prob), …]
    """
    console.print(Panel(
        f"[{THEME['prompt']}]{prompt_text}[/]\n"
        f"[{THEME['dim']}]Enter up to [bold]{n}[/] tokens, one per line.\n"
        f"Format: [bold]token[/]  or  [bold]token : -0.45[/]  (log-prob score).\n"
        f"Type [bold]END[/] when done.[/]",
        border_style="blue",
        title="[bold blue]Token Expansion[/]",
    ))
    tokens: List[Tuple[str, float]] = []
    try:
        while len(tokens) < n:
            raw = input().strip()
            if raw.upper() == "END" or raw == "":
                break
            if ":" in raw:
                parts = raw.split(":", 1)
                tok = parts[0].strip()
                try:
                    lp = float(parts[1].strip())
                except ValueError:
                    lp = 0.0
            else:
                tok = raw
                lp = 0.0
            if tok:
                tokens.append((tok, lp))
    except EOFError:
        pass
    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# Human-input JSONL logger
# ──────────────────────────────────────────────────────────────────────────────

class HumanInputLogger:
    """
    Appends every human oracle interaction to a JSONL file.
    Enables replay, debugging, and future fine-tuning on human preferences.

    Each line in the file is a self-contained JSON record with a timestamp
    and an ``event`` field describing the interaction type.
    """

    def __init__(self, log_path: str = "human_inputs.jsonl") -> None:
        self.log_path = log_path
        if log_path:
            os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".",
                        exist_ok=True)

    def _write(self, record: dict) -> None:
        if not self.log_path:
            return
        record["timestamp"] = datetime.utcnow().isoformat()
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def log_golden_solution(
        self, problem_idx: int, rollout_idx: int, prompt: str,
        teacher_name: str, solution: str, selected: bool,
    ) -> None:
        self._write({
            "event": "golden_solution",
            "problem_idx": problem_idx,
            "rollout_idx": rollout_idx,
            "prompt_preview": prompt[:300],
            "teacher": teacher_name,
            "solution": solution,
            "selected": selected,
        })

    def log_token_expansion(
        self, problem_idx: int, rollout_idx: int, step: int,
        teacher_name: str, tokens: List[Tuple[str, float]],
    ) -> None:
        self._write({
            "event": "token_expansion",
            "problem_idx": problem_idx,
            "rollout_idx": rollout_idx,
            "step": step,
            "teacher": teacher_name,
            "tokens": [[t, lp] for t, lp in tokens],
        })

    def log_process_reward(
        self, problem_idx: int, rollout_idx: int, step: int,
        teacher_name: str, score: float,
    ) -> None:
        self._write({
            "event": "process_reward",
            "problem_idx": problem_idx,
            "rollout_idx": rollout_idx,
            "step": step,
            "teacher": teacher_name,
            "score": score,
        })

    def log_teacher_completion(
        self, problem_idx: int, rollout_idx: int, partial_idx: int,
        teacher_name: str, completion: str,
    ) -> None:
        self._write({
            "event": "teacher_completion",
            "problem_idx": problem_idx,
            "rollout_idx": rollout_idx,
            "partial_idx": partial_idx,
            "teacher": teacher_name,
            "completion": completion,
        })

    def log_student_action(
        self, problem_idx: int, rollout_idx: int, step: int,
        state: str, action: str, logprob: float, is_yes: bool,
    ) -> None:
        self._write({
            "event": "student_action",
            "problem_idx": problem_idx,
            "rollout_idx": rollout_idx,
            "step": step,
            "state": state,
            "action": action,
            "logprob": logprob,
            "is_yes": is_yes,
        })


# ──────────────────────────────────────────────────────────────────────────────
# MCTS nodes
# ──────────────────────────────────────────────────────────────────────────────

class Node:
    def __init__(self, token_id, generated_text: str, parent: Optional["Node"]):
        self.token_id     = token_id
        self.generated_text = generated_text
        self.parent       = parent
        self.children:    List[Node] = []
        self.visit_count  = 0
        self.value        = 0.0

    def add_child(self, child: "Node") -> None:
        self.children.append(child)

    def ucb(self) -> float:
        c = 1.414
        if self.visit_count > 0:
            return (self.value / self.visit_count +
                    c * math.sqrt(math.log(self.parent.visit_count) / self.visit_count))
        return float("inf")


# ──────────────────────────────────────────────────────────────────────────────
# Human-oracle MCTS environment
# ──────────────────────────────────────────────────────────────────────────────

class HumanMCTSEnvironment:
    """
    A fully interactive MCTS environment.
    Every call that previously hit a teacher-model server now surfaces a rich
    terminal prompt for the human operator to fill in.
    """

    def __init__(
        self,
        cfg: InteractiveTrainConfig,
        writer: Optional[SummaryWriter] = None,
        mcts_step_ctr: Optional[List[int]] = None,
        logger: Optional["HumanInputLogger"] = None,
    ):
        self.cfg = cfg
        self.writer        = writer
        self.mcts_step_ctr = mcts_step_ctr if mcts_step_ctr is not None else [0]
        self.logger        = logger
        self.current_prompt     = ""
        self.current_test       = ""
        self.current_entrypoint = ""
        self.root_node: Optional[Node] = None
        self.golden_solution    = ""
        self.golden_tc:  Optional[str] = None   # e.g. "n log n" from O(n log n)
        self.step_count         = 0
        self.last_leaf_text     = ""   # updated each expansion step; used by rollout
        self._problem_idx       = 0    # set by reset()
        self._rollout_idx       = 0    # set by reset()
        self.step_history:      List[dict] = []   # per-step summary records

    # ── public API ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_code(raw: str) -> str:
        """Strip markdown fences and return clean Python code."""
        code = re.sub(r'```(?:python)?\n?', '', raw)
        code = re.sub(r'```', '', code)
        return code.strip()

    @staticmethod
    def _strip_time_complexity(code: str) -> str:
        """Remove a trailing '# Time [Cc]omplexity: O(...)' comment line from code."""
        return re.sub(
            r'\n?#\s*[Tt]ime\s*[Cc]omplexity\s*:.*$', '', code, flags=re.MULTILINE
        ).strip()

    @staticmethod
    def _parse_time_complexity(code: str) -> Optional[str]:
        """Return the expression inside O(...) from a '# Time Complexity: O(...)' comment."""
        m = re.search(r'#\s*[Tt]ime\s*[Cc]omplexity\s*:\s*O\(([^)]+)\)', code)
        return m.group(1).strip() if m else None

    @staticmethod
    def _tc_numeric(tc_str: Optional[str]) -> float:
        """
        Evaluate O-notation expression at n=100 to get a comparable number.
        Returns inf if tc_str is None or cannot be evaluated.
        """
        if tc_str is None:
            return float("inf")
        expr = (
            tc_str
            .replace("n", "100")
            .replace("log n", "math.log(100)")
            .replace("log(n)", "math.log(100)")
        )
        try:
            return float(eval(expr, {"math": math, "__builtins__": {}}))
        except Exception:
            return float("inf")

    def _run_solution_against_tests(
        self, code: str, test: str, entrypoint: str, label: str
    ) -> Tuple[bool, str, str]:
        """
        Write *code* to a temp file, append the HumanEval check harness,
        run it and return (passed: bool, stdout, stderr).
        """
        os.makedirs("temp_golden_eval", exist_ok=True)
        path = f"temp_golden_eval/{label}.py"
        full = (
            code + "\n\n"
            + test + "\n\n"
            + f"try:\n"
            + f"    check({entrypoint})\n"
            + f"    print('TestResult: PASS')\n"
            + f"except Exception as _e:\n"
            + f"    print(f'TestResult: FAIL: {{_e}}')\n"
        )
        with open(path, "w") as fh:
            fh.write(full)
        try:
            res = subprocess.run(
                ["python", path], capture_output=True, text=True, timeout=10
            )
            out, err = res.stdout.strip(), res.stderr.strip()
        except subprocess.TimeoutExpired:
            out, err = "(timeout)", ""
        finally:
            if os.path.exists(path):
                os.remove(path)
        passed = "TestResult: PASS" in out
        return passed, out, err

    def reset(
        self,
        prompt: str,
        test: str,
        entrypoint: str,
        best_solution: str = "",
        problem_idx: int = 0,
        rollout_idx: int = 0,
    ) -> str:
        self.current_prompt     = prompt
        self.current_test       = test
        self.current_entrypoint = entrypoint
        self.step_count         = 0
        self.last_leaf_text     = ""
        self._problem_idx       = problem_idx
        self._rollout_idx       = rollout_idx
        self.root_node          = Node(None, "", None)
        self.step_history       = []

        section("Golden / Best Solution")
        console.print(Panel(
            Syntax(prompt, "python", theme="monokai", word_wrap=True),
            title="[bold cyan]Problem Prompt[/]",
            border_style="cyan",
        ))

        # ── extract code + time complexity from the dataset best_solution ──────
        raw_code         = self._extract_code(best_solution)
        self.golden_tc   = self._parse_time_complexity(raw_code)          # e.g. "n log n"
        self.golden_solution = self._strip_time_complexity(raw_code)      # pure code only

        tc_display = f"O({self.golden_tc})" if self.golden_tc else "(not found)"
        info(f"Golden solution loaded from dataset  |  Time complexity: {tc_display}")

        # ── log the best solution ─────────────────────────────────────────────
        if self.logger:
            self.logger.log_golden_solution(
                self._problem_idx, self._rollout_idx, prompt,
                "dataset", best_solution, selected=True,
            )

        if not self.golden_solution.strip():
            warn("No golden solution found in dataset — process rewards will be 0.")

        self._show_golden()
        return "Length:0, Agree:False, Value:0.00, Nodes:0. Stop? (Yes/No):"

    def step(self, action_text: str) -> Tuple:
        action = action_text.strip().lower()
        self.step_count += 1

        if "yes" in action or self.step_count >= self.cfg.max_steps:
            return self._terminate_and_evaluate()

        # select & display leaf
        leaf = self._select_leaf()
        self.last_leaf_text = leaf.generated_text   # expose for rollout
        self._show_partial(leaf.generated_text)

        # ── per-teacher token expansion ───────────────────────────────────────
        section("Token Expansion  — provide tokens from each teacher")
        all_tokens: List[Tuple[str, float]] = []
        top_tokens_per_teacher: List[Optional[str]] = []

        for t_idx, t_name in enumerate(self.cfg.teacher_names):
            console.print(Rule(
                f"[{THEME['header']}]Teacher {t_idx+1}/{len(self.cfg.teacher_names)}  —  {t_name}[/]",
                style="yellow",
            ))
            # ── try vLLM endpoint first; fall back to human input ─────────────
            t_toks = _fetch_tokens_vllm(
                leaf.generated_text,
                self.cfg.tokens_per_expand,
                t_idx,          # each teacher has its own IP/port in TEACHER_MODELS
            )
            if t_toks is not None:
                _url = TEACHER_MODELS[t_idx]["base_url"] if t_idx < len(TEACHER_MODELS) else "?"
                info(
                    f"vLLM ({_url}) returned "
                    f"{len(t_toks)} token(s) for {t_name}: "
                    + ", ".join(f"{tok!r} ({lp:+.3f})" for tok, lp in t_toks)
                )
            else:
                t_toks = _read_tokens(
                    f"Top-{self.cfg.tokens_per_expand} expansion tokens from [bold]{t_name}[/] "
                    f"(format: token  or  token : -0.45):",
                    self.cfg.tokens_per_expand,
                )
            if t_toks:
                top_tokens_per_teacher.append(t_toks[0][0])
                all_tokens.extend(t_toks)
            else:
                top_tokens_per_teacher.append(None)
            # ── log token expansion per teacher ──────────────────────────────
            if self.logger:
                self.logger.log_token_expansion(
                    self._problem_idx, self._rollout_idx,
                    self.step_count, t_name, t_toks,
                )

        if not all_tokens:
            all_tokens = [(" ", 0.0)]   # fallback

        # auto-derive agreement: all three teachers' top tokens must match
        valid_tops = [t for t in top_tokens_per_teacher if t is not None]
        teachers_agreement = len(valid_tops) == len(self.cfg.teacher_names) and len(set(valid_tops)) == 1

        agree_style = THEME["ok"] if teachers_agreement else THEME["warn"]
        console.print(
            f"  [{agree_style}]Teachers agree on top token: {teachers_agreement}[/]  "
            + (f"(all → {valid_tops[0]!r})" if teachers_agreement and valid_tops else
               f"(tops: {valid_tops})")
        )

        for (tok_text, lp) in all_tokens:
            child = Node(None, leaf.generated_text + tok_text, leaf)
            leaf.add_child(child)

        # ── step context breakdown (shown before asking for process reward) ────
        self._show_step_context(
            step_num=self.step_count,
            leaf_text=leaf.generated_text,
            all_tokens=all_tokens,
            teachers_agreement=teachers_agreement,
            top_tokens_per_teacher=top_tokens_per_teacher,
        )

        # ── per-teacher process reward (averaged) ─────────────────────────────
        self._show_partial(leaf.generated_text, title="Partial for process reward")
        section("Process Reward  — score from each teacher")
        pr_scores: List[float] = []
        for t_idx, t_name in enumerate(self.cfg.teacher_names):
            score = _read_float(
                f"[Teacher {t_idx+1}/{len(self.cfg.teacher_names)}: {t_name}]  "
                f"Score for this partial [-1.0 … 1.0]:",
                -1.0, 1.0,
            )
            pr_scores.append(score)
            # ── log process reward per teacher ──────────────────────────────
            if self.logger:
                self.logger.log_process_reward(
                    self._problem_idx, self._rollout_idx,
                    self.step_count, t_name, score,
                )

        reward = max(-1.0, min(1.0, sum(pr_scores) / len(pr_scores)))

        # ── record step in history ──────────────────────────────────────────
        self.step_history.append({
            "step":        self.step_count,
            "tokens":      [(t, round(lp, 3)) for t, lp in all_tokens[:6]],
            "agree":       teachers_agreement,
            "pr_scores":   pr_scores,
            "reward":      reward,
            "partial_len": len(leaf.generated_text),
        })

        console.print(
            f"  [{THEME['ok']}]Average process reward:[/]  [bold]{reward:+.3f}[/]  "
            f"(from {[f'{s:+.2f}' for s in pr_scores]})"
        )
        self._backpropagate(leaf, reward)

        # build state observation
        seq_length = len(leaf.generated_text)
        if leaf.parent is not None:
            siblings   = leaf.parent.children
            n_nodes    = len(siblings)
            avg_val    = sum(s.value for s in siblings) / n_nodes if n_nodes else 0.0
        else:
            n_nodes  = len(leaf.children)
            avg_val  = leaf.value

        next_state = (
            f"Length:{seq_length}, Agree:{teachers_agreement}, "
            f"Value:{avg_val:.2f}, Nodes:{n_nodes}. Stop? (Yes/No):"
        )

        # ── mcts per-step TensorBoard ─────────────────────────────────────────
        if self.writer is not None:
            _s = self.mcts_step_ctr[0]
            self.writer.add_scalar("mcts/process_reward",  reward,                     _s)
            self.writer.add_scalar("mcts/teachers_agree",  float(teachers_agreement),  _s)
            self.writer.add_scalar("mcts/partial_length",  float(seq_length),          _s)
            self.writer.add_scalar("mcts/tree_nodes",      float(n_nodes),             _s)
            self.writer.add_scalar("mcts/tokens_provided", float(len(all_tokens)),         _s)
            self.writer.add_scalar("mcts/node_avg_value",  avg_val,                    _s)
            self.mcts_step_ctr[0] += 1

        return next_state, 0.0, False

    # ── terminal evaluation ───────────────────────────────────────────────────

    def _terminate_and_evaluate(self) -> Tuple[float, float]:
        all_leaves  = self._get_all_leaves(self.root_node)
        num_leaves  = len(all_leaves)
        top_3       = self._top_3_leaves(all_leaves)

        section("Terminal Evaluation  — 3 teacher completions per partial")
        # completions[partial_idx] = [comp_t1, comp_t2, comp_t3]
        all_completions: List[List[str]] = []

        for i, node in enumerate(top_3, 1):
            self._show_partial(node.generated_text, title=f"Partial #{i} / {len(top_3)}")
            part_comps = _read_teacher_responses(
                self.cfg.teacher_names,
                action_label=f"Complete Partial #{i}",
                hint="Paste this teacher's full completion of the partial above.",
            )
            all_completions.append(part_comps)
            # ── log teacher completions ───────────────────────────────────
            if self.logger:
                for t_name, comp in zip(self.cfg.teacher_names, part_comps):
                    self.logger.log_teacher_completion(
                        self._problem_idx, self._rollout_idx, i - 1, t_name, comp,
                    )

        # flatten to a single list for test running
        flat_completions: List[str] = [
            c for part in all_completions for c in part
        ]

        # run test cases on all completions
        test_passed_reward = self._run_tests(flat_completions)

        # drift reward across all 9 lengths (3 teachers × 3 partials)
        all_lens = [len(c) for c in flat_completions]
        if len(all_lens) >= 2 and max(all_lens) > min(all_lens):
            mean_l  = sum(all_lens) / len(all_lens)
            var     = sum((x - mean_l) ** 2 for x in all_lens) / len(all_lens)
            std     = math.sqrt(var)
            max_std = (max(all_lens) - min(all_lens)) / 2.0
            normed  = std / max_std
            cr_drift = max(-1.0, min(1.0, 1.0 - 2.0 * normed ** 2))
        else:
            cr_drift = 1.0

        final_cr = cr_drift if test_passed_reward == 1.0 else 0.0
        final_pr = -1.0 if num_leaves <= 3 else 1.0

        self._show_rewards(final_cr, final_pr)

        # ── mcts terminal TensorBoard ─────────────────────────────────────────
        if self.writer is not None:
            _s = self.mcts_step_ctr[0]
            self.writer.add_scalar("mcts/num_leaves",     float(num_leaves),                              _s)
            self.writer.add_scalar("mcts/tree_depth",     float(self.step_count),                         _s)
            self.writer.add_scalar("mcts/drift_reward",   cr_drift,                                       _s)
            self.writer.add_scalar("mcts/test_passed",    1.0 if test_passed_reward == 1.0 else 0.0,      _s)
            self.writer.add_scalar("mcts/final_cr",       final_cr,                                       _s)
            self.writer.add_scalar("mcts/final_pr",       final_pr,                                       _s)
            self.writer.add_scalar("mcts/final_total_r",  final_cr + final_pr,                            _s)
            self.mcts_step_ctr[0] += 1

        return final_cr, final_pr

    # ── test runner ───────────────────────────────────────────────────────────

    def _run_tests(self, completions: List[str]) -> float:
        os.makedirs("temp_interactive", exist_ok=True)
        passed_any = False
        results_table = Table(title="Test Results", box=box.ROUNDED, border_style="cyan")
        results_table.add_column("#",         style="dim",        width=4)
        results_table.add_column("File",      style="white",      width=28)
        results_table.add_column("Output",    style="bright_white")
        results_table.add_column("Status",    style="bold",       width=10)

        for i, code in enumerate(completions, 1):
            path = f"temp_interactive/completion_{i}.py"
            with open(path, "w") as fh:
                fh.write(code)
            try:
                res  = subprocess.run(
                    ["python", path], capture_output=True, text=True, timeout=5,
                )
                out  = res.stdout.strip()
                err  = res.stderr.strip()
                text = out or err or "(no output)"
            except subprocess.TimeoutExpired:
                text = "(timeout)"

            match = re.search(r"Passed\s+(\d+)\s+out\s+of\s+(\d+)", text)
            if match:
                p, t = int(match.group(1)), int(match.group(2))
                ok   = p == t and t > 0
                st   = f"[green]✓ {p}/{t}[/]" if ok else f"[red]✗ {p}/{t}[/]"
                if ok:
                    passed_any = True
            else:
                st = "[yellow]?[/]"

            results_table.add_row(str(i), path, escape(text[:80]), st)
            if os.path.exists(path):
                os.remove(path)

        console.print(results_table)
        return 1.0 if passed_any else -1.0

    # ── MCTS internals ────────────────────────────────────────────────────────

    def _select_leaf(self) -> Node:
        node = self.root_node
        while node.children:
            node = max(node.children, key=lambda n: n.ucb())
        return node

    def _backpropagate(self, node: Node, reward: float) -> None:
        node.visit_count += 1
        node.value       += reward
        if node.parent is not None:
            self._backpropagate(node.parent, reward)

    # ── step context breakdown ────────────────────────────────────────────────

    def _show_step_context(
        self,
        step_num: int,
        leaf_text: str,
        all_tokens: List[Tuple[str, float]],
        teachers_agreement: bool,
        top_tokens_per_teacher: List[Optional[str]],
    ) -> None:
        """
        Concise but complete status panel shown before the process-reward prompt.
        Covers:
          • training progress  (problem / rollout / MCTS step)
          • level guesser summary (role, I/O, current target)
          • history table of all steps completed so far
          • current expansion summary (tokens + consensus)
        """
        # ── training progress strip ───────────────────────────────────────────
        progress_text = (
            f"Problem [bold cyan]#{self._problem_idx + 1}[/]  "
            f"│  Rollout [bold cyan]#{self._rollout_idx + 1}[/]  "
            f"│  MCTS Step [bold yellow]{step_num}[/] / [dim]{self.cfg.max_steps}[/]  "
            f"│  Golden TC: [bold green]"
            f"{f'O({self.golden_tc})' if self.golden_tc else 'unknown'}[/]"
        )

        # ── level guesser summary ─────────────────────────────────────────────
        lg_lines = (
            "[bold white]Level Guesser[/bold white] — the student policy that "
            "decides [bold]Yes[/bold] (stop & evaluate) or [bold]No[/bold] "
            "(keep expanding) at each MCTS step.\n"
            "  [dim]Input :[/dim]  state string  "
            "[italic](Length, TeachersAgree, NodeValue, NodeCount)[/italic]\n"
            "  [dim]Output:[/dim]  Yes / No token + log-prob  "
            "[dim](drives GRPO advantage)[/dim]\n"
            f"  [dim]Target :[/dim]  stop when partial converges to  "
            f"[bold green]O({self.golden_tc or '?'})[/bold green]  "
            f"and teachers agree on top token"
        )

        console.print()
        console.print(Panel(
            f"{progress_text}\n\n{lg_lines}",
            title="[bold cyan]▸ Training Context  &  Level Guesser[/bold cyan]",
            border_style="cyan",
            padding=(0, 2),
        ))

        # ── step history table ────────────────────────────────────────────────
        if self.step_history:
            ht = Table(
                title="Step History",
                box=box.SIMPLE_HEAVY,
                border_style="blue",
                show_lines=False,
            )
            ht.add_column("Step",      style="dim",         width=5)
            ht.add_column("Len",       style="white",       width=6)
            ht.add_column("Agree",     style="bold",        width=7)
            ht.add_column("Reward",    style="bold green",  width=8)
            ht.add_column("PR scores", style="dim white",   width=20)
            ht.add_column("Top tokens (up to 3)", style="dim white")
            for rec in self.step_history:
                agree_cell = "[green]✓[/]" if rec["agree"] else "[yellow]✗[/]"
                pr_str  = "  ".join(f"{s:+.2f}" for s in rec["pr_scores"])
                tok_str = "  ".join(f"{t!r}" for t, _ in rec["tokens"][:3])
                ht.add_row(
                    str(rec["step"]),
                    str(rec["partial_len"]),
                    agree_cell,
                    f"{rec['reward']:+.3f}",
                    pr_str,
                    tok_str,
                )
            console.print(ht)

        # ── current expansion summary ─────────────────────────────────────────
        cur_toks = "  ".join(
            f"[bold]{t!r}[/] ({lp:+.2f})"
            for t, lp in all_tokens[: self.cfg.tokens_per_expand]
        )
        agree_badge = (
            f"[bold green]✓ All agree → {top_tokens_per_teacher[0]!r}[/]"
            if (teachers_agreement and top_tokens_per_teacher and
                top_tokens_per_teacher[0] is not None)
            else f"[bold yellow]✗ No consensus  tops={top_tokens_per_teacher}[/]"
        )
        console.print(Panel(
            f"New tokens : {cur_toks or '(none)'}\n"
            f"Consensus  : {agree_badge}",
            title=f"[bold blue]Step {step_num} — Expansion[/bold blue]",
            border_style="blue",
            padding=(0, 2),
        ))

    @staticmethod
    def _get_all_leaves(node: Node) -> List[Node]:
        if not node.children:
            return [node]
        leaves: List[Node] = []
        for child in node.children:
            leaves.extend(HumanMCTSEnvironment._get_all_leaves(child))
        return leaves

    @staticmethod
    def _top_3_leaves(leaves: List[Node]) -> List[Node]:
        def avg(n: Node) -> float:
            return n.value / n.visit_count if n.visit_count else -float("inf")
        return sorted(leaves, key=avg, reverse=True)[:3]

    # ── display helpers ───────────────────────────────────────────────────────

    def _show_golden(self) -> None:
        if self.golden_solution.strip():
            tc_label = f"O({self.golden_tc})" if self.golden_tc else "unknown"
            console.print(Panel(
                Syntax(self.golden_solution, "python", theme="monokai", word_wrap=True),
                title=f"[bold green]Golden Solution[/]  [yellow]Time complexity: {tc_label}[/]",
                border_style="green",
            ))

    def _show_partial(self, text: str, title: str = "Current Partial") -> None:
        display = text if text.strip() else "(empty)"
        console.print(Panel(
            Syntax(display, "python", theme="dracula", word_wrap=True),
            title=f"[bold blue]{title}[/]",
            border_style="blue",
        ))

    def _show_rewards(self, cr: float, pr: float) -> None:
        t = Table(box=box.SIMPLE_HEAVY, border_style="green")
        t.add_column("Metric",        style="bold white",  width=22)
        t.add_column("Value",         style="bold green",  width=10)
        t.add_column("Interpretation", style="dim white")
        t.add_row("Completion Reward (cr)",
                  f"{cr:+.3f}",
                  "drift tightness × test-pass gate")
        t.add_row("Prune Reward (pr)",
                  f"{pr:+.3f}",
                  "+1 if ≥4 leaves explored, else -1")
        t.add_row("Total",
                  f"{cr + pr:+.3f}",
                  "")
        console.print(Panel(t, title="[bold green]Episode Rewards[/]", border_style="green"))


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_student(
    cfg: InteractiveTrainConfig,
    device: torch.device,
) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    section("Loading Student Model")
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:
        task = prog.add_task(f"Loading {cfg.student_model_id} …", total=None)

        tok = AutoTokenizer.from_pretrained(cfg.student_model_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg.student_model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(device)
        model.train()
        prog.update(task, description="Loading frozen reference copy …")

        ref = AutoModelForCausalLM.from_pretrained(
            cfg.student_model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(device)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        prog.update(task, description="Compiling student with torch.compile …")

        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            info("torch.compile done")
        except Exception as exc:
            warn(f"torch.compile skipped: {exc}")

    mem = torch.cuda.memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    info(f"Student loaded  |  GPU mem: {mem:.2f} GB")
    return model, ref, tok


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset_problems() -> list:
    section("Loading Dataset")
    with Progress(SpinnerColumn(), TextColumn("[bold cyan]{task.description}"),
                  console=console, transient=True) as prog:
        prog.add_task("Fetching HumanEval …", total=None)
        try:
            ds = load_dataset("Pradheep1647/openeval_bs", split="test")
        except Exception as exc:
            warn(f"Primary load failed ({exc}); trying evalplus/humanevalplus …")
            ds = load_dataset("evalplus/humanevalplus", split="test")
    info(f"Loaded {len(ds)} problems")
    return list(ds)


# ──────────────────────────────────────────────────────────────────────────────
# Problem selection
# ──────────────────────────────────────────────────────────────────────────────

def select_problem(problems: list) -> dict:
    """Show the first few problems and let the user pick one (or press Enter for random)."""
    section("Problem Selection")

    t = Table(title="Available Problems (first 10)", box=box.ROUNDED, border_style="cyan")
    t.add_column("#",          style="dim",   width=4)
    t.add_column("entry_point", style="bold cyan", width=28)
    t.add_column("Prompt preview", style="white")

    for i, p in enumerate(problems[:10]):
        preview = p["prompt"].strip().splitlines()[0][:60]
        t.add_row(str(i), p.get("entry_point", "?"), escape(preview))

    console.print(t)

    raw = Prompt.ask(
        f"  [{THEME['prompt']}]Pick a problem index (0-{len(problems)-1}) "
        f"or press Enter for random[/]",
        default="",
    )
    if raw.strip() == "":
        idx = random.randint(0, len(problems) - 1)
        info(f"Random selection → #{idx}")
    else:
        try:
            idx = int(raw.strip()) % len(problems)
        except ValueError:
            idx = 0
    return problems[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Rollout
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_rollout(
    student: AutoModelForCausalLM,
    tok: AutoTokenizer,
    env: HumanMCTSEnvironment,
    prompt: str,
    test: str,
    entrypoint: str,
    best_solution: str,
    cfg: InteractiveTrainConfig,
    device: torch.device,
    rollout_idx: int,
) -> Tuple[List[str], List[str], float, float]:

    section(f"Rollout {rollout_idx + 1}")
    state = env.reset(prompt, test, entrypoint, best_solution=best_solution)
    states:  List[str] = []
    actions: List[str] = []

    step_idx = 0
    while True:
        states.append(state)
        step_idx += 1

        # ── show state to user ──────────────────────────────────────────────
        console.print(Panel(
            f"[{THEME['state']}]{escape(state)}[/]",
            title=f"[bold blue]MCTS State  (step {step_idx})[/]",
            border_style="blue",
        ))

        enc = tok(state, return_tensors="pt", truncation=True, max_length=512).to(device)
        try:
            out_ids = student.generate(
                **enc,
                max_new_tokens=cfg.student_max_new_tokens,
                do_sample=True,
                temperature=cfg.student_temperature,
                pad_token_id=tok.pad_token_id,
                use_cache=True,
            )
        except Exception:
            out_ids = student.generate(
                **enc,
                max_new_tokens=cfg.student_max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                use_cache=False,
            )

        gen_ids = out_ids[0, enc["input_ids"].shape[1]:]
        action  = tok.decode(gen_ids, skip_special_tokens=True).strip()
        actions.append(action)

        console.print(
            f"  [{THEME['ok']}]Student action:[/]  "
            f"[bold white]{escape(action)!r}[/]"
        )

        result = env.step(action)

        if (
            isinstance(result, tuple)
            and len(result) == 2
            and not isinstance(result[0], str)
        ):
            cr, pr = float(result[0]), float(result[1])
            break
        elif isinstance(result, tuple) and len(result) == 3:
            next_state, _, done = result
            if done:
                cr, pr = 0.0, 0.0
                break
            state = next_state
        else:
            cr, pr = 0.0, 0.0
            break

    info(f"Rollout done  steps={len(actions)}  cr={cr:+.3f}  pr={pr:+.3f}")
    return states, actions, cr, pr


# ──────────────────────────────────────────────────────────────────────────────
# Rollout result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RolloutResult:
    """All data produced by a single rollout, used by the GDPO loss."""
    states:               List[str]
    actions:              List[str]
    step_logprobs:        List[float]   # mean per-token log-prob of each action
    cr:                   float         # completion reward
    pr:                   float         # prune reward
    yes_step:             int           # step index where student said "yes" (or max_steps+1)
    yes_decisions:        List[bool]    # per-step yes/no decision
    partial_text_at_yes:  str           # MCTS leaf text when stop was triggered


# ──────────────────────────────────────────────────────────────────────────────
# GDPO reward components
# ──────────────────────────────────────────────────────────────────────────────

def get_stop_timing_reward(yes_steps_list: List[int], max_steps: int) -> List[float]:
    """
    Rewards balanced stopping timing across the group of rollouts.

    Intuition:
    - Too early "yes"  → risk of incomplete / wrong code   → penalty
    - Very late  "yes" → inefficient, shallow exploration  → penalty
    - Stopping around ~40-70 % of max_steps → often best

    Formula (per rollout):
        normalized_position = yes_step / max_steps
        r = exp( -(normalized_position - 0.55)^2 / (2*0.15^2) ) * 2 - 1
        → peaks at +1 near 55 % depth; falls to ~ -1 at extremes
    """
    rewards: List[float] = []
    for step in yes_steps_list:
        norm_pos = min(step / max(max_steps, 1), 1.0)  # clamp to [0, 1]
        deviation = (norm_pos - 0.55) ** 2
        r = math.exp(-deviation / (2 * 0.15 ** 2))
        scaled = (r * 2) - 1          # → [-1, +1]
        rewards.append(max(-1.0, min(1.0, scaled)))
    return rewards


def get_premature_stop_penalty(
    yes_decisions: List[bool],
    partial_texts_at_yes: List[str],
) -> List[float]:
    """
    -1 if stopped but code looks incomplete, up to +0.5 if looks functional.

    Checks: has ``def ``, has ``return`` / ``print(``, length > threshold.
    Returns 0.0 if the rollout did not stop voluntarily (continued to max steps).
    """
    rewards: List[float] = []
    for decided_yes, text in zip(yes_decisions, partial_texts_at_yes):
        if not decided_yes:
            rewards.append(0.0)          # no penalty for forced max-step stop
            continue
        has_def     = "def " in text
        has_return  = "return " in text or "print(" in text
        long_enough = len(text.strip()) > 80
        score = 0.0
        if has_def:     score += 0.4
        if has_return:  score += 0.4
        if long_enough: score += 0.2
        r = score - 0.6                  # range: -0.6 → +0.4
        rewards.append(max(-1.0, min(0.5, r)))
    return rewards


def get_confidence_calibration(
    yes_logprobs: List[float],
    decided_yes: List[bool],
) -> List[float]:
    """
    Per-step reward based on how confident the student was in its decision.

    For "yes" actions: reward high log-prob, penalise very low log-prob.
        r = 2 * logprob - 1   (logprob ≈ 0 → +1 at best; very negative → -1)
    For "no" actions: small bonus for a confident continue.
        r = logprob * 0.5
    """
    rewards: List[float] = []
    for lp, is_yes in zip(yes_logprobs, decided_yes):
        if is_yes:
            r = 2.0 * lp - 1.0
        else:
            r = lp * 0.5
        rewards.append(max(-1.0, min(1.0, r)))
    return rewards


# ──────────────────────────────────────────────────────────────────────────────
# GDPO loss  (GRPO with separate per-component advantage normalisation)
# ──────────────────────────────────────────────────────────────────────────────

def compute_grpo_loss(
    student:        AutoModelForCausalLM,
    ref_model:      AutoModelForCausalLM,
    tok:            AutoTokenizer,
    rollout_results: List[RolloutResult],
    cfg:            InteractiveTrainConfig,
    device:         torch.device,
) -> Tuple[torch.Tensor, dict]:
    """
    GDPO loss: five reward components are each independently normalised
    across the group, then *summed* to form the per-trajectory advantage.

    Components
    ----------
    1. cr          – completion reward (drift tightness × test-pass gate)
    2. pr          – prune reward (+1 ≥ 4 leaves explored, else -1)
    3. stop_timing – Gaussian reward for stopping at ~55 % of max_steps
    4. prem_stop   – penalty for stopping when code is still incomplete
    5. confidence  – per-step logprob calibration, averaged to rollout level
    """

    def norm_adv(vals: List[float]) -> List[float]:
        t = torch.tensor(vals, dtype=torch.float32)
        m, s = t.mean(), t.std(unbiased=False).clamp(min=1e-8)
        return ((t - m) / s).tolist()

    group_cr   = [r.cr for r in rollout_results]
    group_pr   = [r.pr for r in rollout_results]
    yes_steps  = [r.yes_step for r in rollout_results]
    yes_decs   = [any(r.yes_decisions) for r in rollout_results]   # bool: voluntarily stopped?
    par_texts  = [r.partial_text_at_yes for r in rollout_results]

    # per-rollout confidence: mean of per-step calibration rewards
    conf_per_rollout: List[float] = []
    for rr in rollout_results:
        step_confs = get_confidence_calibration(rr.step_logprobs, rr.yes_decisions)
        conf_per_rollout.append(
            sum(step_confs) / max(len(step_confs), 1)
        )

    # ── GDPO: independent normalisation of each reward component ────────────
    adv_cr   = norm_adv(group_cr)
    adv_pr   = norm_adv(group_pr)
    adv_stop = norm_adv(get_stop_timing_reward(yes_steps, cfg.max_steps))
    adv_prem = norm_adv(get_premature_stop_penalty(yes_decs, par_texts))
    adv_conf = norm_adv(conf_per_rollout)

    advantages = [
        a + b + c + d + e
        for a, b, c, d, e in zip(adv_cr, adv_pr, adv_stop, adv_prem, adv_conf)
    ]

    total_loss  = torch.zeros(1, device=device)
    n_steps     = 0
    sum_pg = sum_kl = sum_ent = sum_ratio = 0.0
    adv_vals: List[float] = []

    for traj_idx, rr in enumerate(rollout_results):
        if not rr.states:
            continue
        adv = float(advantages[traj_idx])
        adv_vals.append(adv)

        for state_str, action_str in zip(rr.states, rr.actions):
            full_text  = state_str + " " + action_str
            enc        = tok(full_text,  return_tensors="pt", truncation=True, max_length=512).to(device)
            state_enc  = tok(state_str,  return_tensors="pt", truncation=True, max_length=512).to(device)
            state_len  = state_enc["input_ids"].shape[1]
            action_ids = enc["input_ids"][0, state_len:]
            if action_ids.numel() == 0:
                continue

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out_s = student(**enc)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out_r = ref_model(**enc)

            s = state_len - 1
            e = s + action_ids.numel()
            lp_s = F.log_softmax(out_s.logits[0, s:e, :], dim=-1).gather(
                1, action_ids.unsqueeze(1)).squeeze(1).sum()
            lp_r = F.log_softmax(out_r.logits[0, s:e, :], dim=-1).gather(
                1, action_ids.unsqueeze(1)).squeeze(1).sum()

            ratio   = torch.exp(lp_s - lp_r.detach())
            clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
            pg_loss  = -torch.min(ratio * adv, clipped * adv)
            kl_loss  = cfg.kl_coeff   * (lp_s - lp_r.detach())
            ent_bon  = cfg.entropy_coeff * lp_s

            total_loss = total_loss + pg_loss + kl_loss + ent_bon
            n_steps   += 1
            sum_pg    += pg_loss.item()
            sum_kl    += kl_loss.item()
            sum_ent   += ent_bon.item()
            sum_ratio += ratio.item()

    if n_steps > 0:
        total_loss = total_loss / n_steps

    metrics = {
        "pg_loss":    sum_pg    / max(n_steps, 1),
        "kl_loss":    sum_kl    / max(n_steps, 1),
        "ent_bonus":  sum_ent   / max(n_steps, 1),
        "ratio_mean": sum_ratio / max(n_steps, 1),
        "adv_mean":   sum(adv_vals) / max(len(adv_vals), 1),
        "adv_std":    float(torch.tensor(adv_vals).std(unbiased=False))
                      if len(adv_vals) > 1 else 0.0,
        "n_steps":    n_steps,
        # per-component GDPO advantage means (for TensorBoard)
        "adv_cr":     sum(adv_cr)   / max(len(adv_cr),   1),
        "adv_pr":     sum(adv_pr)   / max(len(adv_pr),   1),
        "adv_stop":   sum(adv_stop) / max(len(adv_stop), 1),
        "adv_prem":   sum(adv_prem) / max(len(adv_prem), 1),
        "adv_conf":   sum(adv_conf) / max(len(adv_conf), 1),
    }
    return total_loss.squeeze(), metrics


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: InteractiveTrainConfig = InteractiveTrainConfig()) -> None:
    banner()
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Device: {device}")

    writer = SummaryWriter(log_dir=cfg.log_dir)
    info(f"TensorBoard → {cfg.log_dir}")

    # ── human-input JSONL logger ───────────────────────────────────────────
    logger = HumanInputLogger(cfg.log_inputs_path) if cfg.log_inputs_path else None
    if logger:
        info(f"Human inputs → {cfg.log_inputs_path}")

    student, ref_model, student_tok = load_student(cfg, device)

    problems = load_dataset_problems()
    if cfg.max_problems:
        problems = problems[: cfg.max_problems]
    random.shuffle(problems)
    info(f"Problems in training set: {len(problems)}  |  G={cfg.group_size}")

    # optimiser
    muon_params  = [p for p in student.parameters() if p.dim() == 2]
    other_params = [p for p in student.parameters() if p.dim() != 2]
    muon_opt  = torch.optim.SGD(muon_params,  lr=1e-3, momentum=0.9, weight_decay=0.1)
    other_opt = AdamW(other_params, lr=cfg.lr, weight_decay=0.1, fused=False)

    total_steps = max(1, len(problems) // cfg.grad_accum)
    muon_sched  = get_cosine_schedule_with_warmup(muon_opt,  cfg.warmup_steps, total_steps)
    other_sched = get_cosine_schedule_with_warmup(other_opt, cfg.warmup_steps, total_steps)

    muon_opt.zero_grad(set_to_none=True)
    other_opt.zero_grad(set_to_none=True)

    global_step  = 0
    accum_count  = 0
    run_loss = run_reward = run_cr = run_pr = 0.0
    run_pg = run_kl = run_ent = run_ratio = run_adv = 0.0
    run_stop = run_prem = run_conf = 0.0
    window_n = 0
    mcts_step_ctr = [0]   # shared across all envs in a run; monotonically increasing

    for prob_idx in range(len(problems)):
        # ── problem selection mode ────────────────────────────────────────────
        if Confirm.ask(
            f"\n  [bold cyan]Problem [{prob_idx+1}/{len(problems)}][/]  "
            f"entry=[bold]{problems[prob_idx].get('entry_point','?')}[/]  "
            f"— train on this one?",
            default=True,
        ):
            problem = problems[prob_idx]
        else:
            alt = select_problem(problems)
            problem = alt

        prompt     = problem["prompt"]
        test_str   = problem.get("test", "")
        entrypoint = problem.get("entry_point", "solution")

        group_trajectories: List[Tuple[List[str], List[str]]] = []
        group_cr: List[float] = []
        group_pr: List[float] = []

        # ── G rollouts ────────────────────────────────────────────────────────
        section(f"Problem {prob_idx+1}  ·  {cfg.group_size} rollouts")

        for g in range(cfg.group_size):
            env = HumanMCTSEnvironment(cfg, writer=writer, mcts_step_ctr=mcts_step_ctr)
            states, actions, cr, pr = run_rollout(
                student, student_tok, env,
                prompt, test_str, entrypoint,
                problem.get("best_solution", ""),
                cfg, device, g,
            )
            group_trajectories.append((states, actions))
            group_cr.append(cr)
            group_pr.append(pr)
            _rg = prob_idx * cfg.group_size + g
            _n_yes = sum(1 for a in actions if "yes" in a.lower())
            writer.add_scalar("rollout/cr",      cr,            _rg)
            writer.add_scalar("rollout/pr",      pr,            _rg)
            writer.add_scalar("rollout/total_r", cr + pr,       _rg)
            writer.add_scalar("rollout/steps",   len(actions),  _rg)
            writer.add_scalar("rollout/n_yes",   _n_yes,        _rg)

        if all(len(t[0]) == 0 for t in group_trajectories):
            warn("All rollouts empty — skipping.")
            continue

        # ── GRPO loss + backward ──────────────────────────────────────────────
        section("GRPO Loss")
        try:
            loss, metrics = compute_grpo_loss(
                student, ref_model, student_tok,
                group_trajectories, group_cr, group_pr,
                cfg, device,
            )
            (loss / cfg.grad_accum).backward()
        except Exception as exc:
            warn(f"Loss failed: {exc}")
            muon_opt.zero_grad(set_to_none=True)
            other_opt.zero_grad(set_to_none=True)
            continue

        mean_cr = sum(group_cr) / len(group_cr)
        mean_pr = sum(group_pr) / len(group_pr)

        run_loss   += loss.item()
        run_reward += mean_cr + mean_pr
        run_cr     += mean_cr
        run_pr     += mean_pr
        run_pg     += metrics["pg_loss"]
        run_kl     += metrics["kl_loss"]
        run_ent    += metrics["ent_bonus"]
        run_ratio  += metrics["ratio_mean"]
        run_adv    += metrics["adv_mean"]
        run_stop   += metrics["adv_stop"]
        run_prem   += metrics["adv_prem"]
        run_conf   += metrics["adv_conf"]
        window_n   += 1
        accum_count += 1

        # per-problem TensorBoard (full parity with level_guesser_trainer)
        writer.add_scalar("problem/loss",       loss.item(),            prob_idx)
        writer.add_scalar("problem/mean_r",     mean_cr + mean_pr,      prob_idx)
        writer.add_scalar("problem/mean_cr",    mean_cr,                prob_idx)
        writer.add_scalar("problem/mean_pr",    mean_pr,                prob_idx)
        writer.add_scalar("problem/pg_loss",    metrics["pg_loss"],     prob_idx)
        writer.add_scalar("problem/kl_loss",    metrics["kl_loss"],     prob_idx)
        writer.add_scalar("problem/ent_bonus",  metrics["ent_bonus"],   prob_idx)
        writer.add_scalar("problem/ratio_mean", metrics["ratio_mean"],  prob_idx)
        writer.add_scalar("problem/adv_mean",   metrics["adv_mean"],    prob_idx)
        writer.add_scalar("problem/adv_std",    metrics["adv_std"],     prob_idx)
        writer.add_scalar("problem/n_steps",    metrics["n_steps"],     prob_idx)
        writer.add_scalar("problem/adv_cr",     metrics["adv_cr"],      prob_idx)
        writer.add_scalar("problem/adv_pr",     metrics["adv_pr"],      prob_idx)
        writer.add_scalar("problem/adv_stop",   metrics["adv_stop"],    prob_idx)
        writer.add_scalar("problem/adv_prem",   metrics["adv_prem"],    prob_idx)
        writer.add_scalar("problem/adv_conf",   metrics["adv_conf"],    prob_idx)

        # ── metrics table ─────────────────────────────────────────────────────
        mt = Table(box=box.SIMPLE_HEAVY, border_style="magenta")
        mt.add_column("Metric",  style="bold white", width=18)
        mt.add_column("Value",   style="bold magenta", width=12)
        mt.add_row("Loss",       f"{loss.item():.4f}")
        mt.add_row("PG loss",    f"{metrics['pg_loss']:.4f}")
        mt.add_row("KL loss",    f"{metrics['kl_loss']:.4f}")
        mt.add_row("Ent bonus",  f"{metrics['ent_bonus']:.4f}")
        mt.add_row("Ratio mean", f"{metrics['ratio_mean']:.4f}")
        mt.add_row("Adv mean",   f"{metrics['adv_mean']:+.4f}")
        mt.add_row("mean cr",    f"{mean_cr:+.3f}")
        mt.add_row("mean pr",    f"{mean_pr:+.3f}")
        mt.add_row("adv_stop",   f"{metrics['adv_stop']:+.4f}")
        mt.add_row("adv_prem",   f"{metrics['adv_prem']:+.4f}")
        mt.add_row("adv_conf",   f"{metrics['adv_conf']:+.4f}")
        console.print(Panel(mt, title="[bold magenta]Problem Metrics[/]", border_style="magenta"))

        # ── optimizer step ─────────────────────────────────────────────────────
        if accum_count >= cfg.grad_accum:
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
            muon_opt.step()
            other_opt.step()
            muon_sched.step()
            other_sched.step()
            muon_opt.zero_grad(set_to_none=True)
            other_opt.zero_grad(set_to_none=True)
            global_step += 1
            accum_count  = 0

            avg_loss   = run_loss   / window_n
            avg_reward = run_reward / window_n
            cur_lr     = other_sched.get_last_lr()[0]

            step_t = Table(title=f"Optimizer Step {global_step}", box=box.HEAVY_HEAD,
                           border_style="green")
            step_t.add_column("Metric", style="bold white")
            step_t.add_column("Value",  style="bold green")
            step_t.add_row("Avg loss",   f"{avg_loss:.4f}")
            step_t.add_row("Avg reward", f"{avg_reward:+.4f}")
            step_t.add_row("Avg cr",     f"{run_cr/window_n:+.4f}")
            step_t.add_row("Avg pr",     f"{run_pr/window_n:+.4f}")
            step_t.add_row("Avg stop",   f"{run_stop/window_n:+.4f}")
            step_t.add_row("Avg prem",   f"{run_prem/window_n:+.4f}")
            step_t.add_row("Avg conf",   f"{run_conf/window_n:+.4f}")
            step_t.add_row("LR",         f"{cur_lr:.2e}")
            console.print(Panel(step_t,
                                title=f"[bold green]=== Step {global_step} ===[/]",
                                border_style="green"))

            writer.add_scalar("train/loss",       avg_loss,              global_step)
            writer.add_scalar("train/reward",     avg_reward,            global_step)
            writer.add_scalar("train/cr",         run_cr    / window_n,  global_step)
            writer.add_scalar("train/pr",         run_pr    / window_n,  global_step)
            writer.add_scalar("train/pg_loss",    run_pg    / window_n,  global_step)
            writer.add_scalar("train/kl_loss",    run_kl    / window_n,  global_step)
            writer.add_scalar("train/ent_bonus",  run_ent   / window_n,  global_step)
            writer.add_scalar("train/ratio_mean", run_ratio / window_n,  global_step)
            writer.add_scalar("train/adv_mean",   run_adv   / window_n,  global_step)
            writer.add_scalar("train/adv_stop",   run_stop  / window_n,  global_step)
            writer.add_scalar("train/adv_prem",   run_prem  / window_n,  global_step)
            writer.add_scalar("train/adv_conf",   run_conf  / window_n,  global_step)
            writer.add_scalar("train/lr",         cur_lr,                global_step)

            run_loss = run_reward = run_cr = run_pr = 0.0
            run_pg = run_kl = run_ent = run_ratio = run_adv = 0.0
            run_stop = run_prem = run_conf = 0.0
            window_n = 0

        # ── checkpoint ────────────────────────────────────────────────────────
        if (prob_idx + 1) % cfg.save_every == 0:
            ckpt = os.path.join(cfg.save_dir, f"step_{global_step:06d}")
            student.save_pretrained(ckpt)
            student_tok.save_pretrained(ckpt)
            info(f"Checkpoint → {ckpt}")

    # flush remaining grads
    if accum_count > 0:
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        muon_opt.step();  other_opt.step()
        muon_sched.step(); other_sched.step()
        muon_opt.zero_grad(set_to_none=True)
        other_opt.zero_grad(set_to_none=True)
        global_step += 1

    final = os.path.join(cfg.save_dir, "final")
    student.save_pretrained(final)
    student_tok.save_pretrained(final)
    console.print()
    console.print(Panel(
        f"[bold green]Training complete![/]\n"
        f"Final model saved → [cyan]{final}[/]\n"
        f"Total optimizer steps: [bold]{global_step}[/]",
        title="[bold green]Done[/]",
        border_style="green",
    ))
    writer.close()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Interactive AlphaD-RL trainer (human as oracle)"
    )
    ap.add_argument("--model",       default="Qwen/Qwen3-4B",  help="Student model ID")
    ap.add_argument("--group-size",  type=int, default=2,      help="Rollouts per problem")
    ap.add_argument("--max-problems",type=int, default=80,     help="Max problems to train on")
    ap.add_argument("--max-steps",   type=int, default=8,      help="Max MCTS steps per rollout")
    ap.add_argument("--tokens-per-expand", type=int, default=5,help="Tokens per expansion prompt")
    ap.add_argument("--save-every",  type=int, default=5,      help="Checkpoint interval")
    ap.add_argument("--log-dir",     default="runs/alphaD_rl")
    ap.add_argument("--save-dir",    default="checkpoints")
    args = ap.parse_args()

    cfg = InteractiveTrainConfig(
        student_model_id      = args.model,
        group_size            = args.group_size,
        max_problems          = args.max_problems,
        max_steps             = args.max_steps,
        tokens_per_expand     = args.tokens_per_expand,
        save_every            = args.save_every,
        log_dir               = args.log_dir,
        save_dir              = args.save_dir,
    )
    train(cfg)
