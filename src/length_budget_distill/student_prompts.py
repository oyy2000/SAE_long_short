"""Shared prompts used for student SFT and evaluation."""


def build_student_math_prompt(question: str) -> str:
    return (
        f"Problem:\n{question}\n\n"
        "Solve the problem and end with a line in the form: Answer: <final answer>."
    )


def is_choice_question(question: dict) -> bool:
    """Use the reviewed task type when available, without reading its answer."""
    spec = question['reviewed_answer'] if 'reviewed_answer' in question else question['answer_spec']
    return spec['kind'] == 'choice'


def build_unified_evaluation_prompt(question: dict, config: dict, *, ratio: float | None = None) -> str:
    """Keep choice-letter requirements and TokenSkip conditioning explicit.

    Gold answers and references never enter the prompt. The choice instruction
    is shared by the base student and every baseline, including TokenSkip.
    """
    from .compression_baselines import tokenskip_prompt
    text = question['question']
    if is_choice_question(question):
        text += '\n' + config['choice_instruction']
    if ratio is not None:
        if ratio not in config['tokenskip_ratios']: raise ValueError('Unregistered inference compression ratio')
        return tokenskip_prompt(text,ratio)
    return config['question_template'].format(question=text)
