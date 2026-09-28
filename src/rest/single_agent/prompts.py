import re
from typing import Optional

from mas_prompt import build_code_solver_prompt

REFINED_SLOT = "<<LATENT_REFINED_SLOT>>"


def _code_interface_instruction(task_type: str, fn_name: Optional[str] = None) -> str:
    mode = str(task_type or "").strip().lower()
    if mode in {"function", "functional"}:
        if fn_name:
            return f"Implement and return the function `{fn_name}` only."
        return "Implement and return the required function only."
    return "Write a complete program that reads from stdin and prints to stdout."


def build_code_solver_prompt_no_slot(
    question: str,
    task_type: str,
    fn_name: Optional[str] = None,
) -> str:
    interface = _code_interface_instruction(task_type, fn_name=fn_name)
    final_instruction = (
        "Solve the problem and put the final code inside one markdown code block, "
        "for example ```python\\n<your solution code>\\n```."
    )
    return (
        "You are a solver agent in a multi-agent coding system.\n"
        f"{interface}\n"
        "\n---\nThe programming problem is:\n"
        f"{question}\n"
        f"{final_instruction}"
    )


def build_code_solver_prompt_with_slots(
    question: str,
    task_type: str,
    fn_name: Optional[str] = None,
    args=None,
) -> str:
    return build_code_solver_prompt(question, REFINED_SLOT, task_type, args=args, fn_name=fn_name)


def _math_final_instruction(question: str) -> str:
    if re.search(r"(?mi)^\s*[A-D]\s*[\.\):\-]\s+", question):
        return "Solve the question and put the final choice inside \\boxed{}, for example \\boxed{A}."
    return "Solve the question given information and put the final answer inside \\boxed{}, for example \\boxed{1}."


def build_math_solver_prompt(
    question: str,
    refined_plan: str,
    args=None,
) -> str:
    final_instruction = _math_final_instruction(question)
    if args is not None and args.solver_pre_question == 1:
        return (
            "You are a solver agent in a multi-agent system.\n"
            "The question is:\n"
            "Question:\n"
            f"{question}\n"
            "Here is the refined plan:\n"
            "Refined Plan:\n"
            f"{refined_plan}\n"
            f"{final_instruction}"
        )
    return (
        "You are a solver agent in a multi-agent system.\n"
        "Here is the refined plan:\n"
        "Refined Plan:\n"
        f"{refined_plan}\n"
        "The question is:\n"
        "Question:\n"
        f"{question}\n\n"
        f"{final_instruction}"
    )


def build_math_solver_prompt_with_slots(
    question: str,
    args=None,
) -> str:
    return build_math_solver_prompt(question, REFINED_SLOT, args)


def build_math_solver_prompt_no_slot(
    question: str,
    args=None,
) -> str:
    final_instruction = _math_final_instruction(question)
    if args is not None and args.solver_pre_question == 1:
        return (
            "You are a solver agent in a multi-agent system.\n"
            "The question is:\n"
            "Question:\n"
            f"{question}\n"
            f"{final_instruction}"
        )
    return (
        "You are a solver agent in a multi-agent system.\n"
        "The question is:\n"
        "Question:\n"
        f"{question}\n\n"
        f"{final_instruction}"
    )
