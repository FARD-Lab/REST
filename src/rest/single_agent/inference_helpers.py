from __future__ import annotations

import gc
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from model import resolve_local_pretrained_path
from transformers import AutoModelForCausalLM, AutoTokenizer


def ensure_chat_template(tokenizer, agent_name: str) -> None:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            f"{agent_name} tokenizer does not implement apply_chat_template. "
            "MAS inference requires chat template and does not support fallback."
        )


def load_agent_tokenizer(model_name_or_path: str, trust_remote_code: bool, agent_name: str):
    resolved_path = resolve_local_pretrained_path(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=trust_remote_code, use_fast=True)
    ensure_chat_template(tokenizer, agent_name)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise RuntimeError(f"{agent_name} tokenizer has no pad token and no eos token.")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_agent_model_and_tokenizer(
    model_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
    agent_name: str,
):
    resolved_path = resolve_local_pretrained_path(model_name_or_path)
    tokenizer = load_agent_tokenizer(resolved_path, trust_remote_code=trust_remote_code, agent_name=agent_name)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        torch_dtype=(dtype if dtype != "auto" else "auto"),
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def release_resources(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_inner_adapter(adapter, hidden_states: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    adapter_param = next(adapter.parameters(), None)
    adapter_dtype = adapter_param.dtype if adapter_param is not None else hidden_states.dtype
    x = hidden_states
    if x.dtype != adapter_dtype:
        x = x.to(adapter_dtype)
    out = adapter(x)
    if out.dtype != output_dtype:
        out = out.to(output_dtype)
    return out


def run_outer_adapter(adapter, hidden_states: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    adapter_param = next(adapter.parameters(), None)
    adapter_dtype = adapter_param.dtype if adapter_param is not None else hidden_states.dtype
    x = hidden_states
    if x.dtype != adapter_dtype:
        x = x.to(adapter_dtype)
    out = adapter(x)
    if out.dtype != output_dtype:
        out = out.to(output_dtype)
    return out


@torch.no_grad()
def autoregressive_latent_rollout(
    model,
    rollout_inner_adapter,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_steps: int,
) -> torch.Tensor:
    if latent_steps <= 0:
        raise ValueError("latent_steps must be positive for latent rollout.")

    try:
        decoder = model.get_decoder()
    except (AttributeError, NotImplementedError):
        decoder = getattr(model, "model", None)

    def _last_token_hidden(embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if decoder is not None:
            out = decoder(inputs_embeds=embeds, attention_mask=mask, use_cache=False, return_dict=True)
            hidden = out.last_hidden_state
        else:
            forward_kwargs = {
                "inputs_embeds": embeds,
                "attention_mask": mask,
                "output_hidden_states": True,
                "use_cache": False,
                "return_dict": True,
            }
            try:
                outputs = model(logits_to_keep=1, **forward_kwargs)
            except TypeError:
                outputs = model(**forward_kwargs)
            hidden = outputs.hidden_states[-1]
        return hidden[:, -1, :].clone()

    hidden_states: List[torch.Tensor] = []
    for _ in range(latent_steps):
        last_hidden = _last_token_hidden(input_embeds, attention_mask)
        hidden_states.append(last_hidden.unsqueeze(1))

        next_embed = run_inner_adapter(rollout_inner_adapter, last_hidden, output_dtype=input_embeds.dtype).unsqueeze(1)
        input_embeds = torch.cat([input_embeds, next_embed], dim=1)

        next_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([attention_mask, next_mask], dim=1)

    return torch.cat(hidden_states, dim=1)


def batch_iter_indices(total: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def pad_left_embeds(embed_seqs: Sequence[torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if not embed_seqs:
        raise ValueError("embed_seqs is empty")
    hidden_size = embed_seqs[0].size(-1)
    dtype = embed_seqs[0].dtype
    max_len = max(seq.size(0) for seq in embed_seqs)
    if max_len == 0:
        raise ValueError("Encountered empty embedding sequence.")
    bs = len(embed_seqs)
    batch_embeds = torch.zeros((bs, max_len, hidden_size), dtype=dtype, device=device)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(embed_seqs):
        if seq.size(0) == 0:
            raise ValueError("Found empty embedding sequence; cannot build valid attention mask.")
        seq = seq.to(device=device, dtype=dtype)
        length = seq.size(0)
        batch_embeds[i, max_len - length :, :] = seq
        attention_mask[i, max_len - length :] = 1
    if not torch.all(attention_mask.sum(dim=1) > 0):
        raise ValueError("Invalid padded embed batch: at least one sample has all-zero attention mask.")
    return batch_embeds, attention_mask


def build_generation_kwargs(
    tokenizer,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
) -> Dict[str, object]:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    kwargs: Dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = int(top_k)
        if min_p is not None:
            kwargs["min_p"] = float(min_p)
    if repetition_penalty is not None and float(repetition_penalty) > 0:
        kwargs["repetition_penalty"] = float(repetition_penalty)
    return kwargs
