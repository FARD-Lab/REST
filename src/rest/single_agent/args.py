from __future__ import annotations

import argparse

from rest.recipe import STYLES

SINGLE_AGENT_AXIOMS = ("causality", "minimality", "separability", "stability")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, required=True, choices=STYLES)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--solver_pre_question", type=int, default=0)
    parser.add_argument("--enable_thinking", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gradient_checkpointing", type=int, default=1, choices=[0, 1])

    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_latent_tokens", type=int, default=80)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--outer_lr", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--num_recursive_rounds",
        type=int,
        default=1,
        help="Number of Solver-to-Solver feedback rounds after the initial no-slot pass.",
    )
    parser.add_argument(
        "--axiom",
        type=str,
        nargs="+",
        default=["none"],
        choices=["none", *SINGLE_AGENT_AXIOMS],
        help="Property loss term(s) added at every Solver-to-Solver handoff. Pair each name with a value in --axiom_weight.",
    )
    parser.add_argument(
        "--axiom_weight",
        type=float,
        nargs="+",
        default=[0.0],
        help="Weight (beta) of each --axiom entry, in the same order.",
    )

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--outer_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint-<step> directory written by --save_steps (outer_self.pt plus "
            "training_state.pt). Resumes adapter weights, optimizer, scheduler, and global_step "
            "from there instead of starting at step 0."
        ),
    )
    return parser.parse_args(argv)
