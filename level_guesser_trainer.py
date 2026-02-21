"""
level_guesser_trainer.py

Trains a Qwen/Qwen3-4B model (the "level guesser") with Group Relative Policy
Optimization (GRPO) to decide—at each MCTS step—whether to continue tree
expansion or to stop and evaluate the best partial solutions found so far.

Architecture
------------
* Student  : Qwen/Qwen3-4B   → action policy (Yes / No at each MCTS step)
* Teacher 1: openai/gpt-oss-20b
* Teacher 2: Qwen/Qwen2.5-Coder-7B-Instruct
* Teacher 3: deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct

All models are loaded via Hugging Face AutoModelForCausalLM with explicit
device placement (no device_map="auto") and a shared per-process GPU memory
fraction (mem_fraction=0.25).

Training loop (GRPO)
--------------------
For every coding problem sampled from HumanEval:
  1. Run G independent rollouts under the current student policy.
     Each rollout is an episode through MCTSEnvironment (sequential Yes/No actions).
  2. Collect the terminal reward (completion_reward + prune_reward) for every rollout.
  3. Normalise rewards within the group → group-relative advantages.
  4. Compute PPO-clip surrogate + KL-penalty + entropy-bonus loss.
  5. Accumulate gradients over `grad_accum` problems, then take one optimiser step.
"""

import os
import math
import random
import logging
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset

from mcts_env import MCTSEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # ── student ──────────────────────────────────────────────────────────────
    student_model_id: str   = "Qwen/Qwen3-4B"
    student_max_new_tokens: int = 5       # "Yes" / "No" needs very few tokens
    student_temperature: float = 0.9

    # ── GRPO hypers ───────────────────────────────────────────────────────────
    group_size: int   = 4                   # G rollouts per problem
    clip_eps: float   = 0.2                 # PPO clip ε
    kl_coeff: float   = 0.01               # KL(π_θ ‖ π_ref) penalty weight
    entropy_coeff: float = 0.01            # entropy bonus weight
    lr: float         = 2e-5
    warmup_steps: int = 20
    grad_clip: float  = 1.0
    grad_accum: int   = 4                  # gradient accumulation steps

    # ── training loop ─────────────────────────────────────────────────────────
    max_problems: int = 500                # cap dataset size (None = full)
    save_every: int   = 50                 # checkpoint every N problems
    save_dir: str     = "checkpoints"
    log_dir:  str     = "runs/alphaD_rl"   # TensorBoard log directory

    # ── teacher generation ────────────────────────────────────────────────────
    teacher_temperature: float    = 0.7
    teacher_max_new_tokens: int   = 1024


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────
def _build_gen_params(cfg: TrainConfig) -> dict:
    """Generation kwargs passed directly to HF model.generate()."""
    return {
        "temperature": cfg.teacher_temperature,
        "top_p": 1.0,
        "max_new_tokens": cfg.teacher_max_new_tokens,
    }


def load_teachers(cfg: TrainConfig, device: torch.device):
    """
    Load all three teacher models via Hugging Face.

    Returns:
        hf_tm1, hf_tm2, hf_tm3   - HF AutoModelForCausalLM instances (eval mode)
        tok1, tok2, tok3          - HF AutoTokenizer instances
        params1, params2, params3 - generation kwarg dicts for model.generate()
    """
    model_ids = [
        "openai/gpt-oss-20b",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    ]

    hf_models:  list = []
    tokenizers: list = []

    for mid in model_ids:
        log.info(f"Loading teacher HF model: {mid}")
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        hf = AutoModelForCausalLM.from_pretrained(
            mid,
            trust_remote_code=True,
            dtype=torch.bfloat16,          # fixed: was torch_dtype=
        ).to(device)
        hf.eval()
        tokenizers.append(tok)
        hf_models.append(hf)

    params = [_build_gen_params(cfg)] * 3

    return (
        *hf_models,
        *tokenizers,
        *params,
    )


def load_student(
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    """
    Returns (student, ref_model, tokenizer).
    ref_model is a frozen copy used for the KL penalty in GRPO.
    """
    log.info(f"Loading student model: {cfg.student_model_id}")

    tok = AutoTokenizer.from_pretrained(
        cfg.student_model_id, trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.student_model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,              # fixed: was torch_dtype=
    ).to(device)
    model.train()  # set the model in training mode (for dropout, etc)

    # Frozen reference copy — same weights, no grad
    log.info("Loading frozen reference copy of student for KL penalty …")
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.student_model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,              # fixed: was torch_dtype=
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    return model, ref_model, tok


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def load_coding_dataset() -> list:
    """
    Load HumanEval from the Hugging Face Hub.
    Each example exposes: prompt, test, entry_point.
    """
    log.info("Loading HumanEval dataset …")
    try:
        ds = load_dataset(
            "openai/openai_humaneval", split="test"   # removed trust_remote_code
        )
    except Exception as exc:
        log.warning(f"Primary dataset load failed ({exc}); trying evalplus/humanevalplus …")
        ds = load_dataset(
            "evalplus/humanevalplus", split="test"    # removed trust_remote_code
        )
    log.info(f"Dataset loaded: {len(ds)} problems")
    return list(ds)


# ──────────────────────────────────────────────────────────────────────────────
# Rollout (one episode)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_rollout(
    student: AutoModelForCausalLM,
    tok: AutoTokenizer,
    env: MCTSEnvironment,
    prompt: str,
    test: str,
    entrypoint: str,
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[List[str], List[str], float, float]:
    """
    Run one episode through MCTSEnvironment under the current student policy.

    The student sees a state string such as:
        "Length:42, Agree:True, Value:0.81, Nodes:5. Stop? (Yes/No):"
    and generates a short response ("Yes" or "No").

    Returns
    -------
    states   : state strings encountered during the episode
    actions  : action strings generated by the student
    final_cr : completion reward from MCTSEnvironment._terminate_and_evaluate
    final_pr : prune reward
    """
    state: str = env.reset(prompt, test, entrypoint)
    states:  List[str] = []
    actions: List[str] = []

    while True:
        states.append(state)

        # Tokenise state and sample from the student
        enc = tok(
            state,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        output_ids = student.generate(
            **enc,
            max_new_tokens=cfg.student_max_new_tokens,
            do_sample=True,
            temperature=cfg.student_temperature,
            pad_token_id=tok.pad_token_id,
        )

        # Extract only newly generated tokens
        gen_ids     = output_ids[0, enc["input_ids"].shape[1]:]
        action_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
        actions.append(action_text)

        result = env.step(action_text)

        # MCTSEnvironment.step returns:
        #   • (next_state_str, 0.0, False)   — intermediate step
        #   • (final_cr, final_pr)           — terminal step (2-tuple of floats)
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and not isinstance(result[0], str)
        ):
            # Terminal: (completion_reward, prune_reward)
            final_cr, final_pr = float(result[0]), float(result[1])
            break
        elif isinstance(result, tuple) and len(result) == 3:
            next_state, _intermediate, done = result
            if done:
                final_cr, final_pr = 0.0, 0.0
                break
            state = next_state
        else:
            log.warning(f"Unexpected env.step return: {result!r}")
            final_cr, final_pr = 0.0, 0.0
            break

    return states, actions, final_cr, final_pr


# ──────────────────────────────────────────────────────────────────────────────
# GRPO loss
# ──────────────────────────────────────────────────────────────────────────────

def compute_grpo_loss(
    student:    AutoModelForCausalLM,
    ref_model:  AutoModelForCausalLM,
    tok:        AutoTokenizer,
    group_trajectories: List[Tuple[List[str], List[str]]],
    group_cr:      List[float],
    group_pr:      List[float],
    cfg:    TrainConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, dict]:
    """
    GRPO loss for one group (G trajectories from the same problem).

    For each (state, action) pair across all trajectories:

        ratio   = exp(log π_θ(a|s) - log π_ref(a|s))
        A_i     = (R_i - mean_G) / (std_G + ε)           [group-relative advantage]
        L_pg    = -min(ratio·A_i, clip(ratio, 1±ε)·A_i)  [PPO-clip surrogate]
        L_kl    = β · (log π_θ - log π_ref)               [KL penalty]
        L_ent   = -α · (-log π_θ)                         [entropy bonus, maximised]
        L_step  = L_pg + L_kl + L_ent

    Total loss = mean over all steps in the group.

    Adapting GDPO over here, we normalize then sum the reward, instead of sum then normalize.
    """
    # rewards = torch.tensor(group_rewards, dtype=torch.float32)
    # mean_r  = rewards.mean()
    # std_r   = rewards.std(unbiased=False).clamp(min=1e-8)
    # advantages = ((rewards - mean_r) / std_r).tolist()   # shape: [G]
    cr = torch.tensor(group_cr, dtype=torch.float32)
    pr = torch.tensor(group_pr, dtype=torch.float32)
    mean_cr = cr.mean()
    std_cr = cr.std(unbiased=False).clamp(min=1e-8)
    mean_pr = pr.mean()
    std_pr = pr.std(unbiased=False).clamp(min=1e-8)

    advantages_cr = ((cr - mean_cr) / std_cr).tolist()   # shape: [G]
    advantages_pr = ((pr - mean_pr) / std_pr).tolist()   # shape: [G]

    advantages = [a_cr + a_pr for a_cr, a_pr in zip(advantages_cr, advantages_pr)]

    total_loss   = torch.zeros(1, device=device, requires_grad=False)
    n_steps      = 0
    sum_pg_loss  = 0.0
    sum_kl_loss  = 0.0
    sum_ent      = 0.0
    sum_ratio    = 0.0
    adv_values   = []

    for traj_idx, (states, actions) in enumerate(group_trajectories):
        if not states:
            continue
        adv = float(advantages[traj_idx])
        adv_values.append(adv)

        for state_str, action_str in zip(states, actions):
            full_text  = state_str + " " + action_str
            enc        = tok(full_text,  return_tensors="pt",
                             truncation=True, max_length=512).to(device)
            state_enc  = tok(state_str,  return_tensors="pt",
                             truncation=True, max_length=512).to(device)

            state_len  = state_enc["input_ids"].shape[1]
            action_ids = enc["input_ids"][0, state_len:]   # [n_action_tokens]
            if action_ids.numel() == 0:
                continue

            # ── student forward (grad required) ──────────────────────────────
            out_s      = student(**enc)
            logits_s   = out_s.logits   # [1, seq_len, vocab]

            # ── reference forward (no grad) ───────────────────────────────────
            with torch.no_grad():
                out_r    = ref_model(**enc)
                logits_r = out_r.logits

            # Gather logit slices that predict the action tokens.
            # Token at position t is predicted by logits[:, t-1, :].
            # The first action token is at index state_len, so its logits
            # are at index state_len-1.
            s = state_len - 1
            e = s + action_ids.numel()

            lp_s_all  = F.log_softmax(logits_s[0, s:e, :],  dim=-1)
            lp_r_all  = F.log_softmax(logits_r[0, s:e, :],  dim=-1)

            act_ids_2d = action_ids.unsqueeze(1)                      # [n, 1]
            lp_s = lp_s_all.gather(1, act_ids_2d).squeeze(1).sum()   # scalar
            lp_r = lp_r_all.gather(1, act_ids_2d).squeeze(1).sum()   # scalar (no grad)

            # PPO-clip surrogate
            ratio   = torch.exp(lp_s - lp_r.detach())
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
            pg_loss = -torch.min(ratio * adv, clipped * adv)

            # KL penalty    KL(π_θ ‖ π_ref) ≈ log π_θ − log π_ref
            kl_loss = cfg.kl_coeff * (lp_s - lp_r.detach())

            # Entropy bonus (maximise entropy → subtract negative entropy)
            ent_bonus = cfg.entropy_coeff * lp_s   # lp_s < 0, so this reduces loss

            step_loss  = pg_loss + kl_loss + ent_bonus
            total_loss = total_loss + step_loss
            n_steps   += 1
            sum_pg_loss += pg_loss.item()
            sum_kl_loss += kl_loss.item()
            sum_ent     += ent_bonus.item()
            sum_ratio   += ratio.item()

    if n_steps > 0:
        total_loss = total_loss / n_steps

    metrics = {
        "pg_loss":      sum_pg_loss / max(n_steps, 1),
        "kl_loss":      sum_kl_loss / max(n_steps, 1),
        "ent_bonus":    sum_ent     / max(n_steps, 1),
        "ratio_mean":   sum_ratio   / max(n_steps, 1),
        "adv_mean":     float(sum(adv_values) / max(len(adv_values), 1)),
        "adv_std":      float(torch.tensor(adv_values).std(unbiased=False).item())
                        if len(adv_values) > 1 else 0.0,
        "adv_max":      float(max(adv_values)) if adv_values else 0.0,
        "adv_min":      float(min(adv_values)) if adv_values else 0.0,
        "n_steps":      n_steps,
    }

    return total_loss.squeeze(), metrics

# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────
def train(cfg: TrainConfig = TrainConfig()) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training device: {device}")

    writer = SummaryWriter(log_dir=cfg.log_dir)
    log.info(f"TensorBoard logs → {cfg.log_dir}")

    (
        hf_tm1, hf_tm2, hf_tm3,
        tok1,   tok2,   tok3,
        params1, params2, params3,
    ) = load_teachers(cfg, device)

    student, ref_model, student_tok = load_student(cfg, device)

    problems = load_coding_dataset()
    if cfg.max_problems:
        problems = problems[: cfg.max_problems]
    random.shuffle(problems)
    log.info(f"Training on {len(problems)} problems  |  G={cfg.group_size}")

    # ── optimiser + cosine LR schedules (one per optimizer) ──────────────────
    muon_params  = [p for p in student.parameters() if p.dim() == 2]
    other_params = [p for p in student.parameters() if p.dim() != 2]

    muon_optimizer  = torch.optim.SGD(muon_params,  lr=1e-3, momentum=0.9,
                                      weight_decay=0.1)   # Muon not in stock torch; use SGD
    other_optimizer = AdamW(other_params, lr=cfg.lr, weight_decay=0.1)

    total_opt_steps = max(1, math.ceil(len(problems) / cfg.grad_accum))

    muon_scheduler = get_cosine_schedule_with_warmup(
        muon_optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_opt_steps,
    )
    other_scheduler = get_cosine_schedule_with_warmup(
        other_optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_opt_steps,
    )

    # ── shared MCTS environment ───────────────────────────────────────────────
    env = MCTSEnvironment(
        hf_tm1, hf_tm2, hf_tm3,
        tok1,   tok2,   tok3,
        params1, params2, params3,
    )

    muon_optimizer.zero_grad()
    other_optimizer.zero_grad()
    global_step    = 0
    accum_count    = 0
    running_loss   = 0.0
    running_reward = 0.0
    window_n       = 0
    running_pg     = 0.0
    running_kl     = 0.0
    running_ent    = 0.0
    running_ratio  = 0.0
    running_adv    = 0.0
    running_cr     = 0.0
    running_pr     = 0.0

    for prob_idx, problem in enumerate(problems):
        prompt     = problem["prompt"]
        test_str   = problem.get("test", "")
        entrypoint = problem.get("entry_point", "solution")

        log.info(
            f"[{prob_idx + 1:>4}/{len(problems)}]  entry_point={entrypoint!r}"
        )

        # ── collect G rollouts ────────────────────────────────────────────────
        group_trajectories: List[Tuple[List[str], List[str]]] = []
        group_cr:      List[float] = []
        grpo_pr :      List[float] = []

        for g in range(cfg.group_size):
            try:
                states, actions, cr, pr = run_rollout(
                    student, student_tok, env,
                    prompt, test_str, entrypoint,
                    cfg, device,
                )
                # changing to gdpo, so need to treat them as seperate
                # total_r = cr + pr
                group_cr.append(cr)
                grpo_pr.append(pr)
                group_trajectories.append((states, actions))
                # group_rewards.append(total_r)
                rollout_global = prob_idx * cfg.group_size + g
                writer.add_scalar("rollout/cr",    cr,           rollout_global)
                writer.add_scalar("rollout/pr",    pr,           rollout_global)
                writer.add_scalar("rollout/total_r", cr + pr,    rollout_global)
                writer.add_scalar("rollout/steps", len(actions), rollout_global)
                log.info(
                    f"    rollout {g + 1}/{cfg.group_size}  "
                    f"steps={len(actions):>2}  cr={cr:+.3f}  "
                    f"pr={pr:+.3f}"
                )
            except Exception as exc:
                log.warning(f"    rollout {g + 1} failed: {exc}")
                group_trajectories.append(([], []))
                group_cr.append(0.0)
                grpo_pr.append(0.0)

        # Skip if every rollout failed
        if all(len(t[0]) == 0 for t in group_trajectories):
            log.warning("    All rollouts failed — skipping problem.")
            continue

        # ── compute GRPO loss ─────────────────────────────────────────────────
        try:
            loss, loss_metrics = compute_grpo_loss(
                student, ref_model, student_tok,
                group_trajectories, group_cr, grpo_pr,
                cfg, device,
            )
            # Scale for gradient accumulation before backward
            (loss / cfg.grad_accum).backward()
        except Exception as exc:
            log.warning(f"    Loss computation failed: {exc}")
            muon_optimizer.zero_grad()
            other_optimizer.zero_grad()
            continue

        mean_cr = sum(group_cr) / len(group_cr)
        mean_pr = sum(grpo_pr) / len(grpo_pr)
        mean_r  = mean_cr + mean_pr
        running_loss   += loss.item()
        running_reward += mean_r
        running_pg     += loss_metrics["pg_loss"]
        running_kl     += loss_metrics["kl_loss"]
        running_ent    += loss_metrics["ent_bonus"]
        running_ratio  += loss_metrics["ratio_mean"]
        running_adv    += loss_metrics["adv_mean"]
        running_cr     += mean_cr
        running_pr     += mean_pr
        window_n       += 1
        accum_count    += 1

        # per-problem TensorBoard scalars
        writer.add_scalar("problem/loss",     loss.item(), prob_idx)
        writer.add_scalar("problem/mean_cr",  mean_cr,     prob_idx)
        writer.add_scalar("problem/mean_pr",  mean_pr,     prob_idx)
        writer.add_scalar("problem/mean_r",   mean_r,      prob_idx)
        writer.add_scalar("problem/pg_loss",  loss_metrics["pg_loss"],   prob_idx)
        writer.add_scalar("problem/kl_loss",  loss_metrics["kl_loss"],   prob_idx)
        writer.add_scalar("problem/ent_bonus",loss_metrics["ent_bonus"],  prob_idx)
        writer.add_scalar("problem/ratio_mean",loss_metrics["ratio_mean"],prob_idx)
        writer.add_scalar("problem/adv_mean", loss_metrics["adv_mean"],   prob_idx)
        writer.add_scalar("problem/adv_std",  loss_metrics["adv_std"],    prob_idx)
        writer.add_scalar("problem/adv_max",  loss_metrics["adv_max"],    prob_idx)
        writer.add_scalar("problem/adv_min",  loss_metrics["adv_min"],    prob_idx)

        # ── gradient step after every grad_accum problems ─────────────────────
        if accum_count >= cfg.grad_accum:
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
            muon_optimizer.step()
            other_optimizer.step()
            muon_scheduler.step()    # step both schedulers
            other_scheduler.step()
            muon_optimizer.zero_grad()
            other_optimizer.zero_grad()
            global_step += 1
            accum_count  = 0

            avg_loss   = running_loss   / window_n
            avg_reward = running_reward / window_n
            avg_pg     = running_pg     / window_n
            avg_kl     = running_kl     / window_n
            avg_ent    = running_ent    / window_n
            avg_ratio  = running_ratio  / window_n
            avg_adv    = running_adv    / window_n
            avg_cr     = running_cr     / window_n
            avg_pr     = running_pr     / window_n
            cur_lr     = other_scheduler.get_last_lr()[0]  # use other_scheduler for lr logging

            log.info(
                f"  === optimizer step {global_step}  "
                f"loss={avg_loss:.4f}  "
                f"avg_R={avg_reward:+.4f}  "
                f"lr={cur_lr:.2e} ==="
            )

            writer.add_scalar("train/loss",      avg_loss,   global_step)
            writer.add_scalar("train/reward",    avg_reward, global_step)
            writer.add_scalar("train/cr",        avg_cr,     global_step)
            writer.add_scalar("train/pr",        avg_pr,     global_step)
            writer.add_scalar("train/pg_loss",   avg_pg,     global_step)
            writer.add_scalar("train/kl_loss",   avg_kl,     global_step)
            writer.add_scalar("train/ent_bonus", avg_ent,    global_step)
            writer.add_scalar("train/ratio_mean",avg_ratio,  global_step)
            writer.add_scalar("train/adv_mean",  avg_adv,    global_step)
            writer.add_scalar("train/lr",        cur_lr,     global_step)

            running_loss   = 0.0
            running_reward = 0.0
            running_pg     = 0.0
            running_kl     = 0.0
            running_ent    = 0.0
            running_ratio  = 0.0
            running_adv    = 0.0
            running_cr     = 0.0
            running_pr     = 0.0
            window_n       = 0

        # ── periodic checkpoint ───────────────────────────────────────────────
        if (prob_idx + 1) % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.save_dir, f"step_{global_step:06d}")
            student.save_pretrained(ckpt_path)
            student_tok.save_pretrained(ckpt_path)
            log.info(f"  Checkpoint saved → {ckpt_path}")

    # ── flush any remaining accumulated gradients ─────────────────────────────
    if accum_count > 0:
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        muon_optimizer.step()
        other_optimizer.step()
        muon_scheduler.step()
        other_scheduler.step()
        muon_optimizer.zero_grad()
        other_optimizer.zero_grad()
        global_step += 1

    # ── final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(cfg.save_dir, "final")
    student.save_pretrained(final_path)
    student_tok.save_pretrained(final_path)
    log.info(f"Training complete.  Final model saved → {final_path}")
    writer.close()

if __name__ == "__main__":
    train()
