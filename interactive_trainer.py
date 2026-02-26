"""
interactive_trainer.py

Human-in-the-loop GRPO trainer.

Instead of calling remote teacher-model servers, every oracle call is routed
back to *you* at the terminal:

  • Best solution  → you paste the golden code
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
from typing import List, Optional, Tuple

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

    # ── teacher names (display only) ─────────────────────────────────────────
    teacher_names: List[str] = field(default_factory=lambda: [
        "GPT-oss-20B",
        "Qwen2.5-Coder-14B",
        "Codestral-22B",
    ])


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

    def __init__(self, cfg: InteractiveTrainConfig):
        self.cfg = cfg
        self.current_prompt   = ""
        self.current_test     = ""
        self.current_entrypoint = ""
        self.root_node: Optional[Node] = None
        self.golden_solution  = ""
        self.step_count       = 0

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self, prompt: str, test: str, entrypoint: str) -> str:
        self.current_prompt     = prompt
        self.current_test       = test
        self.current_entrypoint = entrypoint
        self.step_count         = 0
        self.root_node          = Node(None, "", None)

        section("Golden / Best Solution")
        console.print(Panel(
            Syntax(prompt, "python", theme="monokai", word_wrap=True),
            title="[bold cyan]Problem Prompt[/]",
            border_style="cyan",
        ))

        console.print()
        console.print(f"  [{THEME['dim']}]We need the reference / golden solution for process-reward scoring.[/]")
        self.golden_solution = _read_multiline(
            "Paste the BEST / GOLDEN solution for this problem:"
        )
        if not self.golden_solution.strip():
            warn("No golden solution provided — process rewards will be 0.")

        self._show_golden()
        return "Length:0, Agree:False, Value:0.00, Nodes:0. Stop? (Yes/No):"

    def step(self, action_text: str) -> Tuple:
        action = action_text.strip().lower()
        self.step_count += 1

        if "yes" in action or self.step_count >= self.cfg.max_steps:
            return self._terminate_and_evaluate()

        # select & display leaf
        leaf = self._select_leaf()
        self._show_partial(leaf.generated_text)

        # human provides tokens for expansion
        tokens = _read_tokens(
            f"Provide next tokens to EXPAND from this partial (up to {self.cfg.tokens_per_expand}):",
            self.cfg.tokens_per_expand,
        )
        if not tokens:
            tokens = [(" ", 0.0)]   # fallback: single space

        for (tok_text, lp) in tokens:
            child = Node(None, leaf.generated_text + tok_text, leaf)
            leaf.add_child(child)

        teachers_agreement = Confirm.ask(
            f"  [{THEME['prompt']}]Do all teachers agree on the top token?[/]",
            default=False,
        )

        # human rates the process reward
        self._show_partial(leaf.generated_text, title="Partial for process reward")
        reward = _read_float(
            "Process reward for this partial [-1.0 … 1.0]:",
            -1.0, 1.0,
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
        return next_state, 0.0, False

    # ── terminal evaluation ───────────────────────────────────────────────────

    def _terminate_and_evaluate(self) -> Tuple[float, float]:
        all_leaves  = self._get_all_leaves(self.root_node)
        num_leaves  = len(all_leaves)
        top_3       = self._top_3_leaves(all_leaves)

        section("Terminal Evaluation  — Complete the partial solutions")
        completions: List[str] = []

        for i, node in enumerate(top_3, 1):
            self._show_partial(node.generated_text, title=f"Partial #{i}")
            code = _read_multiline(f"Paste the COMPLETION for partial #{i} (test cases will run):")
            completions.append(code)

        # run test cases
        test_passed_reward = self._run_tests(completions)
        lengths = [(len(c),) for c in completions]

        # drift reward
        all_lens = [l for (l,) in lengths]
        if len(all_lens) >= 2:
            mean_l = sum(all_lens) / len(all_lens)
            var    = sum((x - mean_l) ** 2 for x in all_lens) / len(all_lens)
            std    = math.sqrt(var)
            rng    = max(all_lens) - min(all_lens)
            max_std = rng / 2.0 if rng > 0 else 1.0
            normed  = std / max_std
            cr_drift = max(-1.0, min(1.0, 1.0 - 2.0 * normed ** 2))
        else:
            cr_drift = 1.0

        final_cr = cr_drift if test_passed_reward == 1.0 else 0.0
        final_pr = -1.0 if num_leaves <= 3 else 1.0

        self._show_rewards(final_cr, final_pr)
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
            console.print(Panel(
                Syntax(self.golden_solution, "python", theme="monokai", word_wrap=True),
                title="[bold green]Golden Solution (recorded)[/]",
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
            ds = load_dataset("openai/openai_humaneval", split="test")
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
    cfg: InteractiveTrainConfig,
    device: torch.device,
    rollout_idx: int,
) -> Tuple[List[str], List[str], float, float]:

    section(f"Rollout {rollout_idx + 1}")
    state = env.reset(prompt, test, entrypoint)
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
# GRPO loss  (same maths as level_guesser_trainer.py)
# ──────────────────────────────────────────────────────────────────────────────

def compute_grpo_loss(
    student:   AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    tok:       AutoTokenizer,
    group_trajectories: List[Tuple[List[str], List[str]]],
    group_cr:  List[float],
    group_pr:  List[float],
    cfg:       InteractiveTrainConfig,
    device:    torch.device,
) -> Tuple[torch.Tensor, dict]:

    cr = torch.tensor(group_cr, dtype=torch.float32)
    pr = torch.tensor(group_pr, dtype=torch.float32)

    def norm_adv(t: torch.Tensor) -> List[float]:
        m, s = t.mean(), t.std(unbiased=False).clamp(min=1e-8)
        return ((t - m) / s).tolist()

    advantages = [a + b for a, b in zip(norm_adv(cr), norm_adv(pr))]

    total_loss  = torch.zeros(1, device=device)
    n_steps     = 0
    sum_pg = sum_kl = sum_ent = sum_ratio = 0.0
    adv_vals: List[float] = []

    for traj_idx, (states, actions) in enumerate(group_trajectories):
        if not states:
            continue
        adv = float(advantages[traj_idx])
        adv_vals.append(adv)

        for state_str, action_str in zip(states, actions):
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
        "pg_loss":   sum_pg    / max(n_steps, 1),
        "kl_loss":   sum_kl    / max(n_steps, 1),
        "ent_bonus": sum_ent   / max(n_steps, 1),
        "ratio_mean":sum_ratio / max(n_steps, 1),
        "adv_mean":  sum(adv_vals) / max(len(adv_vals), 1),
        "adv_std":   float(torch.tensor(adv_vals).std(unbiased=False))
                     if len(adv_vals) > 1 else 0.0,
        "n_steps":   n_steps,
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
    window_n = 0

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
            env = HumanMCTSEnvironment(cfg)
            states, actions, cr, pr = run_rollout(
                student, student_tok, env,
                prompt, test_str, entrypoint,
                cfg, device, g,
            )
            group_trajectories.append((states, actions))
            group_cr.append(cr)
            group_pr.append(pr)
            writer.add_scalar("rollout/cr", cr, prob_idx * cfg.group_size + g)
            writer.add_scalar("rollout/pr", pr, prob_idx * cfg.group_size + g)

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
        window_n   += 1
        accum_count += 1

        # per-problem TensorBoard
        writer.add_scalar("problem/loss",   loss.item(), prob_idx)
        writer.add_scalar("problem/pg_loss",metrics["pg_loss"], prob_idx)
        writer.add_scalar("problem/kl_loss",metrics["kl_loss"], prob_idx)
        writer.add_scalar("problem/mean_cr",mean_cr, prob_idx)
        writer.add_scalar("problem/mean_pr",mean_pr, prob_idx)

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
            step_t.add_row("LR",         f"{cur_lr:.2e}")
            console.print(Panel(step_t,
                                title=f"[bold green]=== Step {global_step} ===[/]",
                                border_style="green"))

            writer.add_scalar("train/loss",   avg_loss,   global_step)
            writer.add_scalar("train/reward", avg_reward, global_step)
            writer.add_scalar("train/lr",     cur_lr,     global_step)

            run_loss = run_reward = run_cr = run_pr = 0.0
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
    ap.add_argument("--max-problems",type=int, default=20,     help="Max problems to train on")
    ap.add_argument("--max-steps",   type=int, default=8,      help="Max MCTS steps per rollout")
    ap.add_argument("--tokens-per-expand", type=int, default=5,help="Tokens per expansion prompt")
    ap.add_argument("--save-every",  type=int, default=5,      help="Checkpoint interval")
    ap.add_argument("--log-dir",     default="runs/alphaD_rl_interactive")
    ap.add_argument("--save-dir",    default="checkpoints_interactive")
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
