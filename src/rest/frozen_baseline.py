from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List, Sequence

import torch
from common import render_chat_ids, resolve_dtype
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
from rest.single_agent.inference_helpers import (
    batch_iter_indices,
    build_generation_kwargs,
    load_agent_model_and_tokenizer,
    release_resources,
)
from rest.single_agent.prompts import (
    build_code_solver_prompt_no_slot,
    build_math_solver_prompt_no_slot,
)


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_raw_baseline_stage(
    model,
    tokenizer,
    prompts: Sequence[str],
    enable_thinking: bool,
    batch_size: int,
    device: torch.device,
    gen_kwargs: dict,
    num_rollouts: int,
    on_record=None,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for start, end in batch_iter_indices(len(prompts), batch_size):
        batch_prompts = prompts[start:end]
        batch_n = end - start

        prompt_id_seqs = [
            render_chat_ids(
                tokenizer,
                p,
                assistant_text=None,
                enable_thinking=enable_thinking,
            )
            for p in batch_prompts
        ]
        padded = tokenizer.pad({"input_ids": prompt_id_seqs}, padding=True, return_tensors="pt")
        input_ids = padded["input_ids"].to(device)
        attention_mask = padded["attention_mask"].to(device)

        _sync_cuda()
        t0 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
        _sync_cuda()
        generate_time_per_example = (time.perf_counter() - t0) / max(batch_n, 1)

        prompt_len = input_ids.size(1)
        gen_ids = generated[:, prompt_len:]
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
            record = {"rollouts": rollouts}
            records.append(record)
            if on_record is not None:
                on_record(start + i, record)

    return records


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)

    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--enable_thinking", type=int, default=0, choices=[0, 1])
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
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = resolve_dtype(args.dtype)
    if model_dtype is None:
        raise ValueError("Unsupported dtype configuration.")
    enable_thinking = bool(args.enable_thinking)

    model, tokenizer = load_agent_model_and_tokenizer(
        args.model_name_or_path,
        device=device,
        dtype=model_dtype,
        trust_remote_code=args.trust_remote_code,
        agent_name="raw_baseline",
    )

    is_code = is_code_eval_dataset(args.eval_dataset)

    dataset_arg = resolve_medqa_dataset_arg(args.eval_dataset, RMAS_INFERENCE_DIR)
    dataset_split = args.dataset_split or DATASET_DEFAULT_SPLIT.get(args.eval_dataset.lower(), "test")

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

    if is_code:
        prompts = [
            build_code_solver_prompt_no_slot(
                question=q,
                task_type=sample_metadata[i]["task_type"],
                fn_name=sample_metadata[i]["fn_name"],
            )
            for i, q in enumerate(questions)
        ]
    else:
        prompts = [build_math_solver_prompt_no_slot(q) for q in questions]

    code_eval_timeout_s = args.mbppplus_timeout_s if is_mbppplus_dataset(args.eval_dataset) else args.lcb_timeout_s

    _LIGHT_MODEL_IDS = {"Qwen/Qwen3-1.7B", "meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-Math-1.5B-Instruct"}
    style = "sequential_light" if args.model_name_or_path in _LIGHT_MODEL_IDS else "sequential_scaled"
    max_new_tokens = args.max_new_tokens or infer_max_new_tokens(style, args.eval_dataset)
    temperature = infer_temperature(args.eval_dataset, args.temperature)
    num_rollouts = args.num_rollouts or (10 if args.eval_dataset.lower() in {"aime25", "aime26"} else 1)
    if num_rollouts > 1 and not args.do_sample:
        raise ValueError(
            f"--num_rollouts={num_rollouts} needs sampling, but --do_sample is 0: every rollout "
            "would be identical and pass@k would collapse to pass@1."
        )
    print(
        f"[eval] {args.eval_dataset}: max_new_tokens={max_new_tokens} num_rollouts={num_rollouts} "
        f"do_sample={bool(args.do_sample)} temperature={temperature} top_p={args.top_p} "
        f"batch_size={args.batch_size}",
        flush=True,
    )

    gen_kwargs = build_generation_kwargs(
        tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=bool(args.do_sample),
        temperature=temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
    )
    if num_rollouts > 1:
        gen_kwargs["num_return_sequences"] = num_rollouts

    num_correct = 0
    total_tokens = 0
    total_time_seconds = 0.0
    f = open(args.result_jsonl, "w", encoding="utf-8")

    def on_record(sample_idx, gen):
        nonlocal num_correct, total_tokens, total_time_seconds
        question, gold_text = questions[sample_idx], gold_answers[sample_idx]
        graded = []
        for rollout in gen["rollouts"]:
            if is_code:
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

        sample_tokens = sum(int(g["num_tokens"]) for g in graded)
        sample_time = sum(float(g["time_seconds"]) for g in graded)
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
        f.flush()

    run_raw_baseline_stage(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        enable_thinking=enable_thinking,
        batch_size=args.batch_size,
        device=device,
        gen_kwargs=gen_kwargs,
        num_rollouts=num_rollouts,
        on_record=on_record,
    )

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
    f.close()

    print(f"[eval] {args.eval_dataset}: {num_correct}/{num_examples} correct ({summary['accuracy']:.4f})", flush=True)

    release_resources(model, tokenizer)


if __name__ == "__main__":
    main()
