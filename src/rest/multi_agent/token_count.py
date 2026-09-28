from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class _SolverCall:
    rollout_idx: int
    final_outputs: List[str] = field(default_factory=list)


@dataclass
class TokenCounter:
    num_recursive_rounds: Optional[int] = None
    latent_steps: Optional[int] = None
    num_rollouts: Optional[int] = None
    result_jsonl: Optional[str] = None
    solver_calls: List[_SolverCall] = field(default_factory=list)
    solver_tokenizer: Any = None

    @property
    def latent_total_tokens(self) -> int:
        assert self.num_recursive_rounds is not None and self.latent_steps is not None
        return self.latent_steps * (3 * self.num_recursive_rounds - 1)

    def _call_for_rollout(self, rollout_idx: int) -> _SolverCall:
        if rollout_idx < len(self.solver_calls):
            return self.solver_calls[rollout_idx]
        if not self.solver_calls:
            raise RuntimeError("TokenCounter: no run_solver_latent_stage call was observed.")
        return self.solver_calls[0]

    def num_tokens_for(self, rollout_idx: int, sample_idx: int) -> int:
        if self.num_recursive_rounds is None or self.latent_steps is None:
            raise RuntimeError("TokenCounter: --num_recursive_rounds and --latent_steps were never captured.")
        if self.solver_tokenizer is None:
            raise RuntimeError("TokenCounter: the solver tokenizer was never captured.")
        call = self._call_for_rollout(rollout_idx)
        text = call.final_outputs[sample_idx]
        n_final = len(self.solver_tokenizer(text, add_special_tokens=False)["input_ids"])
        if rollout_idx == 0:
            return self.latent_total_tokens + n_final
        return n_final


def _install_parse_args_patch(mod, counter: TokenCounter) -> None:
    orig = mod.parse_args

    @functools.wraps(orig)
    def patched():
        args = orig()
        counter.num_recursive_rounds = int(args.num_recursive_rounds)
        counter.latent_steps = int(args.latent_steps)
        counter.num_rollouts = int(args.num_rollouts)
        counter.result_jsonl = str(args.result_jsonl or "").strip()
        return args

    mod.parse_args = patched


def _install_tokenizer_capture_patch(mod, counter: TokenCounter) -> None:
    orig = mod.load_agent_model_and_tokenizer

    @functools.wraps(orig)
    def patched(*args, **kwargs):
        model, tokenizer = orig(*args, **kwargs)
        if kwargs.get("agent_name") == "solver":
            counter.solver_tokenizer = tokenizer
        return model, tokenizer

    mod.load_agent_model_and_tokenizer = patched


def _install_solver_stage_patch(mod, counter: TokenCounter) -> None:
    orig = mod.run_solver_latent_stage

    @functools.wraps(orig)
    def patched(*args, **kwargs):
        call = _SolverCall(rollout_idx=len(counter.solver_calls))
        counter.solver_calls.append(call)
        outputs = orig(*args, **kwargs)
        call.final_outputs = list(outputs)
        return outputs

    mod.run_solver_latent_stage = patched


def _install_retry_stage_patch(mod, counter: TokenCounter) -> None:
    orig = mod.run_answer_retry_stage

    @functools.wraps(orig)
    def patched(*args, **kwargs):
        updated_outputs, num_retried = orig(*args, **kwargs)
        if not counter.solver_calls:
            raise RuntimeError("TokenCounter: run_answer_retry_stage ran before any run_solver_latent_stage call.")
        counter.solver_calls[-1].final_outputs = list(updated_outputs)
        return updated_outputs, num_retried

    mod.run_answer_retry_stage = patched


def install(inference_mas_module, counter: TokenCounter) -> None:
    _install_parse_args_patch(inference_mas_module, counter)
    _install_tokenizer_capture_patch(inference_mas_module, counter)
    _install_solver_stage_patch(inference_mas_module, counter)
    _install_retry_stage_patch(inference_mas_module, counter)


def augment_result_jsonl(counter: TokenCounter) -> int:
    path = (counter.result_jsonl or "").strip()
    if not path:
        return 0
    if not os.path.exists(path):
        raise FileNotFoundError(f"TokenCounter: expected result_jsonl at {path!r} but it does not exist.")

    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not counter.solver_calls:
        raise RuntimeError("TokenCounter: no run_solver_latent_stage call was observed during this run.")
    if counter.num_rollouts is not None and len(counter.solver_calls) < counter.num_rollouts:
        print(
            f"[token_count][warn] observed {len(counter.solver_calls)} solver-stage call(s) but "
            f"--num_rollouts={counter.num_rollouts}; later rollouts reuse rollout 0's text."
        )

    augmented = 0
    for record in records:
        if record.get("type") == "summary":
            continue
        sample_idx = record.get("sample_idx")
        if sample_idx is None:
            continue

        rollouts = record.get("rollouts")
        if isinstance(rollouts, list):
            for rollout_record in rollouts:
                rollout_idx = int(rollout_record["rollout_idx"])
                rollout_record["num_tokens"] = counter.num_tokens_for(rollout_idx, sample_idx)
                augmented += 1
        else:
            rollout_idx = int(record.get("rollout_idx", 0))
            record["num_tokens"] = counter.num_tokens_for(rollout_idx, sample_idx)
            augmented += 1

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return augmented
