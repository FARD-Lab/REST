from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import torch
from common import (
    build_stage_with_slot,
    compute_solver_ce_loss,
    load_inner_adapter,
    load_model_and_tokenizer,
    load_outer_training_dataset,
    render_chat_ids,
    resolve_dtype,
    run_inner_adapter_preserve_input_grad,
    run_outer_adapter,
    trim_latent,
)
from model import CrossModelAdapter
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from rest.axioms import AxiomTerm, HandoffContext, build_axiom_term
from rest.recipe import (
    BASE_MODELS,
    INNER_ADAPTER_TYPE_FALLBACK,
    OUTER_ADAPTER_TYPE,
    TRAIN_DATASET,
    TRAIN_SPLIT,
    inner_adapter_path,
)
from rest.single_agent.args import parse_args
from rest.single_agent.checkpoint import (
    cleanup_intermediate_checkpoints,
    load_resume_checkpoint,
    save_selfloop_checkpoint,
)
from rest.single_agent.prompts import (
    REFINED_SLOT,
    build_math_solver_prompt,
    build_math_solver_prompt_no_slot,
    build_math_solver_prompt_with_slots,
)


def activate_gc_runtime(model: torch.nn.Module) -> None:
    model.train()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def build_solver_teacher_forced_inputs(
    tokenizer,
    question: str,
    answer: str,
    solver_args: argparse.Namespace,
    enable_thinking: bool,
    device: torch.device,
    max_length: int,
):
    user_prompt = build_math_solver_prompt_no_slot(question, solver_args)
    prompt_ids = render_chat_ids(
        tokenizer, user_prompt, assistant_text=None, enable_thinking=enable_thinking, max_length=max_length
    )
    full_ids = render_chat_ids(
        tokenizer, user_prompt, assistant_text=answer, enable_thinking=enable_thinking, max_length=max_length
    )

    assistant_token_count = max(len(full_ids) - len(prompt_ids), 0)
    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        assistant_kept = min(assistant_token_count, len(full_ids))
        prompt_len = len(full_ids) - assistant_kept
    else:
        prompt_len = min(len(prompt_ids), len(full_ids))

    assistant_mask = torch.zeros((len(full_ids),), dtype=torch.bool, device=device)
    if prompt_len < len(full_ids):
        assistant_mask[prompt_len:] = True

    input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, assistant_mask


def compute_weighted_axiom_loss(
    axiom_terms: List[Tuple[str, AxiomTerm, float]], ctx: HandoffContext
) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    per_name: Dict[str, torch.Tensor] = {}
    for name, term, weight in axiom_terms:
        value = term.compute(ctx)
        if value is not None:
            per_name[name] = weight * value
    if not per_name:
        return None, per_name
    return torch.stack(list(per_name.values())).sum(), per_name


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.num_recursive_rounds <= 0:
        raise ValueError("--num_recursive_rounds must be positive.")
    if len(args.axiom) != len(args.axiom_weight):
        raise ValueError("--axiom and --axiom_weight must have the same number of values.")
    if "none" in args.axiom and len(args.axiom) != 1:
        raise ValueError("--axiom none cannot be combined with other axiom names.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = resolve_dtype(args.dtype)
    outer_dtype = resolve_dtype(args.outer_dtype)
    if model_dtype is None or outer_dtype is None:
        raise ValueError("Unsupported dtype configuration.")

    if device.type == "cpu" and model_dtype in {torch.float16, torch.bfloat16}:
        print("[warn] CPU + fp16/bf16 is unstable. Falling back model dtype to float32.")
        model_dtype = torch.float32
    if device.type == "cpu" and outer_dtype in {torch.float16, torch.bfloat16}:
        print("[warn] CPU + fp16/bf16 is unstable. Falling back outer dtype to float32.")
        outer_dtype = torch.float32

    torch.manual_seed(args.seed)
    enable_thinking = bool(args.enable_thinking)
    solver_args = argparse.Namespace(solver_pre_question=args.solver_pre_question)

    solver_model, solver_tok = load_model_and_tokenizer(
        BASE_MODELS[args.style]["solver"],
        device=device,
        dtype=model_dtype,
        trust_remote_code=args.trust_remote_code,
        agent_name="solver",
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    if bool(args.gradient_checkpointing):
        activate_gc_runtime(solver_model)

    solver_embed = solver_model.get_input_embeddings()
    solver_hidden = solver_embed.weight.size(-1)

    active_axioms = [(name, weight) for name, weight in zip(args.axiom, args.axiom_weight) if name != "none"]
    composed = len(active_axioms) > 1

    axiom_terms: List[Tuple[str, AxiomTerm, float]] = [
        (
            name,
            build_axiom_term(
                name,
                weight,
                planner_hidden_size=solver_hidden,
                refiner_hidden_size=solver_hidden,
                solver_hidden_size=solver_hidden,
                device=device,
                dtype=outer_dtype,
                composed=composed,
            ),
            weight,
        )
        for name, weight in active_axioms
    ]
    if composed:
        print(f"[axioms] composed run over {[name for name, _ in active_axioms]}", flush=True)

    inner_solver = load_inner_adapter(
        inner_adapter_path(args.style, "solver"),
        hidden_size=solver_hidden,
        device=device,
        dtype=model_dtype,
        fallback_adapter_type=INNER_ADAPTER_TYPE_FALLBACK,
    )

    outer_self = CrossModelAdapter(solver_hidden, solver_hidden, OUTER_ADAPTER_TYPE).to(
        device=device, dtype=outer_dtype
    )
    outer_self.train()

    params = list(outer_self.parameters())
    for _, term, _ in axiom_terms:
        params += list(term.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.outer_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    dataset = load_outer_training_dataset(TRAIN_DATASET, TRAIN_SPLIT, None)
    needed_cols = {"question", "answer"}
    missing = needed_cols.difference(set(dataset.column_names))
    if missing:
        raise ValueError(f"Dataset missing required fields: {sorted(missing)}")
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    rows = [{"question": sample.get("question", ""), "answer": sample.get("answer", "")} for sample in dataset]

    dataloader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: x)
    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty. Increase dataset size or reduce batch_size.")

    steps_per_epoch = len(dataloader)
    max_train_steps = args.max_steps if args.max_steps > 0 else args.num_train_epochs * steps_per_epoch

    if args.lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_train_steps, num_cycles=0.5
        )
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    os.makedirs(args.save_dir, exist_ok=True)

    global_step = 0
    if args.resume_from:
        global_step = load_resume_checkpoint(
            args.resume_from, outer_self, optimizer, scheduler, device, axiom_terms=axiom_terms
        )
        print(f"[resume] resumed from {args.resume_from} at global_step={global_step}", flush=True)
    apply_axiom = bool(axiom_terms)

    log_loss = 0.0
    log_axiom = 0.0
    log_axiom_count = 0
    log_axiom_by_name: Dict[str, float] = {}
    log_axiom_count_by_name: Dict[str, int] = {}
    log_count = 0

    while global_step < max_train_steps:
        for batch in dataloader:
            if global_step >= max_train_steps:
                break

            sample_axiom_losses: List[float] = []
            sample_axiom_losses_by_name: Dict[str, List[float]] = {}
            valid_count = 0
            batch_loss_sum = 0.0

            optimizer.zero_grad(set_to_none=True)

            for sample in batch:
                q = str(sample["question"]).strip()
                ans = str(sample["answer"]).strip()
                if not q or not ans:
                    continue

                try:
                    round_losses: List[torch.Tensor] = []
                    axiom_losses: List[torch.Tensor] = []
                    axiom_losses_by_name: Dict[str, List[torch.Tensor]] = {}

                    input_ids, attention_mask, assistant_mask = build_solver_teacher_forced_inputs(
                        solver_tok, q, ans, solver_args, enable_thinking, device, args.max_length
                    )
                    with torch.no_grad():
                        pass0_out = solver_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            use_cache=False,
                            return_dict=True,
                        )
                    prev_hidden = pass0_out.hidden_states[-1][0][assistant_mask]
                    prev_logits = pass0_out.logits
                    prev_assistant_mask = assistant_mask

                    if prev_hidden.size(0) == 0:
                        continue

                    for round_idx in range(args.num_recursive_rounds):
                        is_final_round = round_idx == args.num_recursive_rounds - 1

                        prev_inner = run_inner_adapter_preserve_input_grad(
                            inner_solver, prev_hidden, out_dtype=model_dtype
                        )
                        feedback_vector = run_outer_adapter(outer_self, prev_inner, out_dtype=solver_embed.weight.dtype)
                        feedback_vector = trim_latent(feedback_vector, args.max_latent_tokens)

                        solver_user_with_slot = build_math_solver_prompt_with_slots(q, args=solver_args)
                        solver_pack = build_stage_with_slot(
                            tokenizer=solver_tok,
                            embedding_layer=solver_embed,
                            user_prompt_with_slot=solver_user_with_slot,
                            assistant_text=ans,
                            slot_text=REFINED_SLOT,
                            slot_embeds=feedback_vector,
                            enable_thinking=enable_thinking,
                            device=device,
                            embed_dtype=solver_embed.weight.dtype,
                            max_length=args.max_length,
                        )
                        solver_out = solver_model(
                            inputs_embeds=solver_pack.inputs_embeds,
                            attention_mask=solver_pack.attention_mask,
                            output_hidden_states=not is_final_round,
                            use_cache=False,
                            return_dict=True,
                        )
                        loss_round = compute_solver_ce_loss(solver_out.logits, solver_pack.labels)
                        if torch.isnan(loss_round) or torch.isinf(loss_round):
                            round_losses = []
                            break
                        round_losses.append(loss_round)

                        if apply_axiom:
                            reference_prompt = build_math_solver_prompt(q, ans, solver_args)
                            producer_input_text = build_math_solver_prompt_no_slot(q, solver_args)
                            ctx = HandoffContext(
                                downstream_model=solver_model,
                                downstream_tokenizer=solver_tok,
                                downstream_embedding_layer=solver_embed,
                                reference_prompt=reference_prompt,
                                upstream_text=ans,
                                downstream_text=ans,
                                vector=feedback_vector,
                                trained_logits=solver_out.logits,
                                trained_assistant_mask=solver_pack.assistant_mask,
                                enable_thinking=enable_thinking,
                                device=device,
                                max_length=args.max_length,
                                embed_dtype=solver_embed.weight.dtype,
                                upstream_logits=prev_logits,
                                upstream_assistant_mask=prev_assistant_mask,
                                upstream_input_text=producer_input_text,
                            )
                            term_value, term_per_name = compute_weighted_axiom_loss(axiom_terms, ctx)
                            if term_value is not None:
                                axiom_losses.append(term_value)
                                for _name, _value in term_per_name.items():
                                    axiom_losses_by_name.setdefault(_name, []).append(_value)

                        if not is_final_round:
                            prev_hidden = solver_out.hidden_states[-1][0][solver_pack.assistant_mask]
                            if prev_hidden.size(0) == 0:
                                round_losses = []
                                break
                            prev_logits = solver_out.logits
                            prev_assistant_mask = solver_pack.assistant_mask

                    if not round_losses:
                        continue

                    loss = round_losses[-1]

                    if axiom_losses:
                        axiom_term_value = torch.stack(axiom_losses).mean()
                        loss = loss + axiom_term_value
                        sample_axiom_losses.append(float(axiom_term_value.item()))
                        for _name, _values in axiom_losses_by_name.items():
                            sample_axiom_losses_by_name.setdefault(_name, []).append(
                                float(torch.stack(_values).mean().item())
                            )

                    (loss / max(args.batch_size, 1)).backward()
                    valid_count += 1
                    batch_loss_sum += float(loss.item())
                except RuntimeError as exc:
                    exc_msg = str(exc).lower()
                    if "sequence_too_long" in exc_msg:
                        continue
                    raise

            if valid_count == 0:
                continue

            if valid_count != args.batch_size:
                grad_scale = args.batch_size / valid_count
                for param in params:
                    if param.grad is not None:
                        param.grad.mul_(grad_scale)

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1
            loss_batch = batch_loss_sum / valid_count

            log_loss += float(loss_batch)
            if sample_axiom_losses:
                log_axiom += sum(sample_axiom_losses) / len(sample_axiom_losses)
                log_axiom_count += 1
            for _name, _values in sample_axiom_losses_by_name.items():
                if _values:
                    log_axiom_by_name[_name] = log_axiom_by_name.get(_name, 0.0) + sum(_values) / len(_values)
                    log_axiom_count_by_name[_name] = log_axiom_count_by_name.get(_name, 0) + 1
            log_count += 1

            if global_step % args.log_every == 0:
                avg_loss = log_loss / max(log_count, 1)
                avg_axiom = log_axiom / max(log_axiom_count, 1)
                avg_axiom_by_name = {
                    _name: log_axiom_by_name[_name] / max(log_axiom_count_by_name[_name], 1)
                    for _name in log_axiom_by_name
                }
                _per_axiom_str = " ".join(f"{_name}={_avg:.4f}" for _name, _avg in avg_axiom_by_name.items())
                print(
                    f"step={global_step} loss={avg_loss:.4f} axiom_loss={avg_axiom:.4f} [{_per_axiom_str}]",
                    flush=True,
                )
                log_loss = 0.0
                log_axiom = 0.0
                log_axiom_count = 0
                log_axiom_by_name = {}
                log_axiom_count_by_name = {}
                log_count = 0

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_selfloop_checkpoint(
                    args.save_dir,
                    global_step,
                    outer_self,
                    args,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    axiom_terms=axiom_terms,
                )

            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    save_selfloop_checkpoint(
        args.save_dir,
        None,
        outer_self,
        args,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        axiom_terms=axiom_terms,
    )
    cleanup_intermediate_checkpoints(args.save_dir)


if __name__ == "__main__":
    main()
