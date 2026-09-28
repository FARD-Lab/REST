from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List, Optional, Sequence

import torch
from common import (
    ids_to_embeds,
    load_inner_adapter,
    render_chat_ids,
    render_chat_text,
    resolve_dtype,
    split_rendered_text_by_slot,
    text_to_ids,
)
from hf_resolver import resolve_medqa_dataset_arg
from inference_utils.answer_utils import compare_answers
from inference_utils.inference_mas import load_eval_questions_and_answers
from inference_utils.lcb_utils import (
    evaluate_generated_code,
    extract_python_code,
    is_code_eval_dataset,
    is_mbppplus_dataset,
)
from load_from_repo import DATASET_DEFAULT_SPLIT
from run import infer_max_new_tokens, infer_temperature

from rest import RMAS_INFERENCE_DIR
from rest.recipe import BASE_MODELS, INNER_ADAPTER_TYPE_FALLBACK, STYLES, inner_adapter_path
from rest.single_agent.checkpoint import load_outer_self_adapter
from rest.single_agent.inference_helpers import (
    autoregressive_latent_rollout,
    batch_iter_indices,
    build_generation_kwargs,
    load_agent_model_and_tokenizer,
    pad_left_embeds,
    release_resources,
    run_inner_adapter,
    run_outer_adapter,
)
from rest.single_agent.prompts import (
    REFINED_SLOT,
    build_code_solver_prompt_no_slot,
    build_code_solver_prompt_with_slots,
    build_math_solver_prompt_no_slot,
    build_math_solver_prompt_with_slots,
)


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_solver_selfloop_stage(
    model,
    tokenizer,
    embed_layer,
    embed_dtype: torch.dtype,
    inner_solver,
    outer_self,
    questions: Sequence[str],
    solver_args: argparse.Namespace,
    enable_thinking: bool,
    latent_steps: int,
    num_recursive_rounds: int,
    batch_size: int,
    device: torch.device,
    gen_kwargs: dict,
    num_rollouts: int,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    latent_total_tokens = latent_steps * num_recursive_rounds
    for start, end in batch_iter_indices(len(questions), batch_size):
        batch_questions = questions[start:end]
        batch_n = end - start

        no_slot_embed_seqs = []
        for q in batch_questions:
            user_prompt = build_math_solver_prompt_no_slot(q, solver_args)
            prompt_ids = render_chat_ids(tokenizer, user_prompt, assistant_text=None, enable_thinking=enable_thinking)
            no_slot_embed_seqs.append(ids_to_embeds(embed_layer, prompt_ids, device=device, dtype=embed_dtype))
        batch_embeds, attention_mask = pad_left_embeds(no_slot_embed_seqs, device=device)

        _t0 = time.perf_counter()
        hidden_rollout = autoregressive_latent_rollout(
            model=model,
            rollout_inner_adapter=inner_solver,
            input_embeds=batch_embeds,
            attention_mask=attention_mask,
            latent_steps=latent_steps,
        )
        _sync_cuda()
        rollout_time_per_example = (time.perf_counter() - _t0) / max(batch_n, 1)

        solver_inner_out = run_inner_adapter(inner_solver, hidden_rollout, output_dtype=embed_dtype)
        feedback_vectors = run_outer_adapter(outer_self, solver_inner_out, output_dtype=embed_dtype)

        with_slot_embed_seqs = []
        for i, q in enumerate(batch_questions):
            user_prompt_with_slot = build_math_solver_prompt_with_slots(q, args=solver_args)
            rendered = render_chat_text(
                tokenizer, user_prompt_with_slot, assistant_text=None, enable_thinking=enable_thinking
            )
            prefix_text, suffix_text = split_rendered_text_by_slot(rendered, REFINED_SLOT)
            prefix_embeds = ids_to_embeds(
                embed_layer, text_to_ids(tokenizer, prefix_text), device=device, dtype=embed_dtype
            )
            suffix_embeds = ids_to_embeds(
                embed_layer, text_to_ids(tokenizer, suffix_text), device=device, dtype=embed_dtype
            )
            vector = feedback_vectors[i]
            if vector.dtype != embed_dtype:
                vector = vector.to(embed_dtype)
            with_slot_embed_seqs.append(torch.cat([prefix_embeds, vector, suffix_embeds], dim=0))
        final_embeds, final_mask = pad_left_embeds(with_slot_embed_seqs, device=device)

        _t0 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(inputs_embeds=final_embeds, attention_mask=final_mask, **gen_kwargs)
        _sync_cuda()
        generate_time_per_example = (time.perf_counter() - _t0) / max(batch_n, 1)

        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        prompt_len = final_mask.size(1)
        if sequences.size(1) > gen_kwargs["max_new_tokens"]:
            gen_ids = sequences[:, prompt_len:]
        else:
            gen_ids = sequences
        batch_texts = [text.strip() for text in tokenizer.batch_decode(gen_ids, skip_special_tokens=True)]

        for i in range(batch_n):
            rollout_texts = batch_texts[i * num_rollouts : (i + 1) * num_rollouts]
            rollouts = []
            for rollout_idx, text in enumerate(rollout_texts):
                n_final = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                rollouts.append(
                    {
                        "rollout_idx": rollout_idx,
                        "generated": text,
                        "num_tokens": n_final,
                        "time_seconds": generate_time_per_example / num_rollouts,
                    }
                )
            records.append(
                {
                    "rollouts": rollouts,
                    "latent_tokens": latent_total_tokens,
                    "latent_time_seconds": rollout_time_per_example,
                }
            )

    return records


def run_solver_selfloop_code_stage(
    model,
    tokenizer,
    embed_layer,
    embed_dtype: torch.dtype,
    inner_solver,
    outer_self,
    questions: Sequence[str],
    task_types: Sequence[str],
    fn_names: Sequence[Optional[str]],
    solver_args: argparse.Namespace,
    enable_thinking: bool,
    latent_steps: int,
    num_recursive_rounds: int,
    batch_size: int,
    device: torch.device,
    gen_kwargs: dict,
    num_rollouts: int,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    latent_total_tokens = latent_steps * num_recursive_rounds
    for start, end in batch_iter_indices(len(questions), batch_size):
        batch_questions = questions[start:end]
        batch_task_types = task_types[start:end]
        batch_fn_names = fn_names[start:end]
        batch_n = end - start

        no_slot_embed_seqs = []
        for q, task_type, fn_name in zip(batch_questions, batch_task_types, batch_fn_names):
            user_prompt = build_code_solver_prompt_no_slot(question=q, task_type=task_type, fn_name=fn_name)
            prompt_ids = render_chat_ids(tokenizer, user_prompt, assistant_text=None, enable_thinking=enable_thinking)
            no_slot_embed_seqs.append(ids_to_embeds(embed_layer, prompt_ids, device=device, dtype=embed_dtype))
        batch_embeds, attention_mask = pad_left_embeds(no_slot_embed_seqs, device=device)

        _t0 = time.perf_counter()
        hidden_rollout = autoregressive_latent_rollout(
            model=model,
            rollout_inner_adapter=inner_solver,
            input_embeds=batch_embeds,
            attention_mask=attention_mask,
            latent_steps=latent_steps,
        )
        _sync_cuda()
        rollout_time_per_example = (time.perf_counter() - _t0) / max(batch_n, 1)

        solver_inner_out = run_inner_adapter(inner_solver, hidden_rollout, output_dtype=embed_dtype)
        feedback_vectors = run_outer_adapter(outer_self, solver_inner_out, output_dtype=embed_dtype)

        with_slot_embed_seqs = []
        for i, (q, task_type, fn_name) in enumerate(zip(batch_questions, batch_task_types, batch_fn_names)):
            user_prompt_with_slot = build_code_solver_prompt_with_slots(q, task_type, fn_name=fn_name, args=solver_args)
            rendered = render_chat_text(
                tokenizer, user_prompt_with_slot, assistant_text=None, enable_thinking=enable_thinking
            )
            prefix_text, suffix_text = split_rendered_text_by_slot(rendered, REFINED_SLOT)
            prefix_embeds = ids_to_embeds(
                embed_layer, text_to_ids(tokenizer, prefix_text), device=device, dtype=embed_dtype
            )
            suffix_embeds = ids_to_embeds(
                embed_layer, text_to_ids(tokenizer, suffix_text), device=device, dtype=embed_dtype
            )
            vector = feedback_vectors[i]
            if vector.dtype != embed_dtype:
                vector = vector.to(embed_dtype)
            with_slot_embed_seqs.append(torch.cat([prefix_embeds, vector, suffix_embeds], dim=0))
        final_embeds, final_mask = pad_left_embeds(with_slot_embed_seqs, device=device)

        _t0 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(inputs_embeds=final_embeds, attention_mask=final_mask, **gen_kwargs)
        _sync_cuda()
        generate_time_per_example = (time.perf_counter() - _t0) / max(batch_n, 1)

        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        prompt_len = final_mask.size(1)
        if sequences.size(1) > gen_kwargs["max_new_tokens"]:
            gen_ids = sequences[:, prompt_len:]
        else:
            gen_ids = sequences
        batch_texts = [text.strip() for text in tokenizer.batch_decode(gen_ids, skip_special_tokens=True)]

        for i in range(batch_n):
            rollout_texts = batch_texts[i * num_rollouts : (i + 1) * num_rollouts]
            rollouts = []
            for rollout_idx, text in enumerate(rollout_texts):
                n_final = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                rollouts.append(
                    {
                        "rollout_idx": rollout_idx,
                        "generated": text,
                        "num_tokens": n_final,
                        "time_seconds": generate_time_per_example / num_rollouts,
                    }
                )
            records.append(
                {
                    "rollouts": rollouts,
                    "latent_tokens": latent_total_tokens,
                    "latent_time_seconds": rollout_time_per_example,
                }
            )

    return records


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, required=True, choices=STYLES)
    parser.add_argument("--outer_checkpoint_dir", type=str, required=True)

    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--solver_pre_question", type=int, default=0)
    parser.add_argument("--enable_thinking", type=int, default=0, choices=[0, 1])
    parser.add_argument("--latent_steps", type=int, default=32)
    parser.add_argument("--num_recursive_rounds", type=int, default=1)
    parser.add_argument("--num_rollouts", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--max_new_tokens", type=int, default=0)
    parser.add_argument("--do_sample", type=int, default=1, choices=[0, 1])
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--outer_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--lcb_use_private_tests", type=int, default=1, choices=[0, 1])
    parser.add_argument("--lcb_timeout_s", type=int, default=6)
    parser.add_argument("--mbppplus_timeout_s", type=int, default=10)
    parser.add_argument("--mbppplus_num_prompt_tests", type=int, default=3)
    parser.add_argument("--mbppplus_subset", type=str, default="")
    parser.add_argument("--mbppplus_cache_dir", type=str, default="")

    parser.add_argument("--result_jsonl", type=str, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.eval_dataset.lower() == "livecodebench":
        args.eval_dataset = "lcb"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = resolve_dtype(args.dtype)
    outer_dtype = resolve_dtype(args.outer_dtype)
    if model_dtype is None or outer_dtype is None:
        raise ValueError("Unsupported dtype configuration.")

    solver_args = argparse.Namespace(solver_pre_question=args.solver_pre_question)
    enable_thinking = bool(args.enable_thinking)

    model, tokenizer = load_agent_model_and_tokenizer(
        BASE_MODELS[args.style]["solver"],
        device=device,
        dtype=model_dtype,
        trust_remote_code=args.trust_remote_code,
        agent_name="solver",
    )
    embed_layer = model.get_input_embeddings()
    embed_dtype = embed_layer.weight.dtype
    solver_hidden = embed_layer.weight.size(-1)

    inner_solver = load_inner_adapter(
        inner_adapter_path(args.style, "solver"),
        hidden_size=solver_hidden,
        device=device,
        dtype=model_dtype,
        fallback_adapter_type=INNER_ADAPTER_TYPE_FALLBACK,
    )
    outer_self = load_outer_self_adapter(
        args.outer_checkpoint_dir,
        hidden_size=solver_hidden,
        device=device,
        dtype=outer_dtype,
    )

    dataset_arg = resolve_medqa_dataset_arg(args.eval_dataset, RMAS_INFERENCE_DIR)
    dataset_split = args.dataset_split or DATASET_DEFAULT_SPLIT.get(args.eval_dataset.lower(), "test")

    is_code_eval = is_code_eval_dataset(args.eval_dataset)
    sample_metadata = None
    task_types = None
    fn_names = None
    if is_code_eval:
        _, questions, gold_answers, sample_metadata = load_eval_questions_and_answers(
            dataset_arg,
            dataset_split,
            args.num_samples,
            args.shuffle,
            args.seed,
            return_metadata=True,
            lcb_use_private_tests=bool(args.lcb_use_private_tests),
            mbppplus_subset=args.mbppplus_subset,
            mbppplus_cache_dir=args.mbppplus_cache_dir,
            mbppplus_num_prompt_tests=args.mbppplus_num_prompt_tests,
        )
        task_types = [str(meta.get("task_type", "complete")) for meta in sample_metadata]
        fn_names = [meta.get("fn_name") if isinstance(meta, dict) else None for meta in sample_metadata]
    else:
        _, questions, gold_answers = load_eval_questions_and_answers(
            dataset_arg,
            dataset_split,
            args.num_samples,
            args.shuffle,
            args.seed,
        )

    max_new_tokens = args.max_new_tokens or infer_max_new_tokens(args.style, args.eval_dataset)
    num_rollouts = args.num_rollouts or (10 if args.eval_dataset.lower() in {"aime25", "aime26"} else 1)
    if num_rollouts > 1 and not args.do_sample:
        raise ValueError(
            f"--num_rollouts={num_rollouts} needs sampling, but --do_sample is 0: every rollout "
            "would be identical and pass@k would collapse to pass@1."
        )

    if is_mbppplus_dataset(args.eval_dataset):
        effective_temperature = infer_temperature(args.eval_dataset, args.temperature)
    elif is_code_eval:
        effective_temperature = 0.2
    else:
        effective_temperature = args.temperature

    code_eval_timeout_s = None
    if is_code_eval:
        code_eval_timeout_s = args.mbppplus_timeout_s if is_mbppplus_dataset(args.eval_dataset) else args.lcb_timeout_s

    print(
        f"[eval] {args.eval_dataset}: max_new_tokens={max_new_tokens} num_rollouts={num_rollouts} "
        f"do_sample={bool(args.do_sample)} temperature={effective_temperature} top_p={args.top_p} "
        f"batch_size={args.batch_size}",
        flush=True,
    )

    gen_kwargs = build_generation_kwargs(
        tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=bool(args.do_sample),
        temperature=effective_temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
    )
    if num_rollouts > 1:
        gen_kwargs["num_return_sequences"] = num_rollouts

    if is_code_eval:
        generations = run_solver_selfloop_code_stage(
            model=model,
            tokenizer=tokenizer,
            embed_layer=embed_layer,
            embed_dtype=embed_dtype,
            inner_solver=inner_solver,
            outer_self=outer_self,
            questions=questions,
            task_types=task_types,
            fn_names=fn_names,
            solver_args=solver_args,
            enable_thinking=enable_thinking,
            latent_steps=args.latent_steps,
            num_recursive_rounds=args.num_recursive_rounds,
            batch_size=args.batch_size,
            device=device,
            gen_kwargs=gen_kwargs,
            num_rollouts=num_rollouts,
        )
    else:
        generations = run_solver_selfloop_stage(
            model=model,
            tokenizer=tokenizer,
            embed_layer=embed_layer,
            embed_dtype=embed_dtype,
            inner_solver=inner_solver,
            outer_self=outer_self,
            questions=questions,
            solver_args=solver_args,
            enable_thinking=enable_thinking,
            latent_steps=args.latent_steps,
            num_recursive_rounds=args.num_recursive_rounds,
            batch_size=args.batch_size,
            device=device,
            gen_kwargs=gen_kwargs,
            num_rollouts=num_rollouts,
        )

    num_correct = 0
    total_tokens = 0
    total_time_seconds = 0.0
    with open(args.result_jsonl, "w", encoding="utf-8") as f:
        for sample_idx, (question, gold_text, gen) in enumerate(zip(questions, gold_answers, generations)):
            graded = []
            for rollout in gen["rollouts"]:
                if is_code_eval:
                    pred_code = extract_python_code(rollout["generated"])
                    eval_result = evaluate_generated_code(
                        pred_code,
                        sample_metadata[sample_idx]["eval_sample"],
                        timeout_s=code_eval_timeout_s,
                    )
                    gold_answer, pred_answer, correct = gold_text, pred_code, bool(eval_result.get("all_passed"))
                else:
                    gold_answer, pred_answer, correct, _, _ = compare_answers(
                        gold_text, rollout["generated"], args.eval_dataset
                    )
                graded.append(
                    {
                        "rollout_idx": rollout["rollout_idx"],
                        "generated": rollout["generated"],
                        "gold_answer": gold_answer,
                        "pred_answer": pred_answer,
                        "correct": correct,
                        "num_tokens": rollout["num_tokens"],
                        "time_seconds": rollout["time_seconds"],
                    }
                )

            sample_tokens = int(gen["latent_tokens"]) + sum(int(g["num_tokens"]) for g in graded)
            sample_time = float(gen["latent_time_seconds"]) + sum(float(g["time_seconds"]) for g in graded)
            total_tokens += sample_tokens
            total_time_seconds += sample_time

            record = {
                "sample_idx": sample_idx,
                "dataset": args.eval_dataset,
                "dataset_split": dataset_split,
                "question": question,
                "gold_text": gold_text,
                "gold_answer": graded[0]["gold_answer"],
                "num_tokens": sample_tokens,
                "time_seconds": sample_time,
                "latent_tokens": gen["latent_tokens"],
            }
            if num_rollouts > 1:
                record["pass_at_k_correct"] = any(g["correct"] for g in graded)
                record["rollouts"] = graded
                num_correct += int(record["pass_at_k_correct"])
            else:
                only = graded[0]
                record["generated"] = only["generated"]
                record["pred_answer"] = only["pred_answer"]
                record["correct"] = only["correct"]
                num_correct += int(only["correct"])
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        num_examples = len(questions)
        summary = {
            "type": "summary",
            "eval_dataset": args.eval_dataset,
            "num_rollouts": num_rollouts,
            "metric": f"pass@{num_rollouts}" if num_rollouts > 1 else "accuracy",
            "num_examples": num_examples,
            "num_correct": num_correct,
            "accuracy": num_correct / max(num_examples, 1),
            "total_tokens": total_tokens,
            "total_time_seconds": total_time_seconds,
        }
        f.write(json.dumps(summary) + "\n")

    print(f"[eval] {args.eval_dataset}: {num_correct}/{num_examples} correct ({summary['accuracy']:.4f})", flush=True)

    release_resources(model, tokenizer, inner_solver, outer_self)


if __name__ == "__main__":
    main()
