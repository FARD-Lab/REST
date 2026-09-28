from __future__ import annotations

import argparse
import sys

from rest import RMAS_INFERENCE_DIR
from rest.multi_agent.run_patch import load_patched_run_module
from rest.multi_agent.token_count import TokenCounter, augment_result_jsonl, install


def _forward_argv(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--outer_dir")
    known, rest = parser.parse_known_args(argv)
    if any(arg == "--ckpt_override" or arg.startswith("--ckpt_override=") for arg in rest):
        raise SystemExit("error: --ckpt_override is not accepted. Pass the trained outer links with --outer_dir.")
    if "-h" in rest or "--help" in rest:
        print("REST option: --outer_dir DIR  training output directory holding the outer links (required).\n")
        return rest
    if not known.outer_dir:
        raise SystemExit("error: --outer_dir is required.")
    return rest + ["--ckpt_override", f"outer={known.outer_dir}"]


def main() -> int:
    sys.argv = [sys.argv[0], *_forward_argv(sys.argv[1:])]
    run_module = load_patched_run_module(RMAS_INFERENCE_DIR / "run.py", "rmas_release_run")

    counter = TokenCounter()
    install(run_module.inference_mas, counter)

    exit_code = run_module.main()

    num_augmented = augment_result_jsonl(counter)
    if counter.result_jsonl:
        print(f"[evaluate] wrote num_tokens into {num_augmented} record(s) of {counter.result_jsonl}")
    else:
        print("[evaluate] no --result_jsonl was passed; nothing to augment.")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
