"""Shared prompts used for student SFT and evaluation."""


def build_student_math_prompt(question: str) -> str:
    return (
        f"Problem:\n{question}\n\n"
        "Solve the problem and end with a line in the form: Answer: <final answer>."
    )
