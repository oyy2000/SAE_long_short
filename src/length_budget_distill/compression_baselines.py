"""Text-side components for published DAP and TokenSkip-style distillation."""
from __future__ import annotations

import hashlib
import re


TOKEN_SKIP_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def ratio_for_problem(problem_id, ratios, seed):
    """One deterministic ratio per question, independent of input row ordering."""
    if not ratios or len(set(ratios)) != len(ratios) or any(not 0 < r <= 1 for r in ratios):
        raise ValueError("Retention ratios must be distinct and in (0, 1]")
    digest = hashlib.sha256(f"{seed}|{problem_id}".encode()).digest()
    return ratios[int.from_bytes(digest[:8], "big") % len(ratios)]


def balanced_ratio_assignment(problem_ids, ratios, seed):
    """Freeze exact count balance across the full question cohort before filtering.

    Correct-source filtering can subsequently unbalance the retained subset;
    never reassign ratios after observing compression quality or eligible support.
    """
    ids=list(problem_ids)
    if len(ids)!=len(set(ids)) or not ids:raise ValueError('Unique nonempty question IDs required')
    ratio_for_problem(ids[0],ratios,seed)  # Reuse registered ratio validation.
    ordered=sorted(ids,key=lambda pid:(hashlib.sha256(f'{seed}|{pid}'.encode()).hexdigest(),pid))
    return {pid:ratios[i%len(ratios)] for i,pid in enumerate(ordered)}


def tokenskip_prompt(question, ratio):
    """Same Qwen content for SFT and inference; 1.0 has no ratio suffix.

    Preserve the upstream literal <|eot_id|> marker even on Qwen: it is text,
    not an added vocabulary token. Chat boundaries come from the tokenizer.
    """
    if not 0 < ratio <= 1:
        raise ValueError("Retention ratio must be in (0, 1]")
    suffix = f"<|eot_id|>{float(ratio):.1f}<|eot_id|>" if ratio < 1 else ""
    return TOKEN_SKIP_INSTRUCTION + "\n" + question + suffix


def tokenskip_completion(compressed_cot, verified_answer):
    return compressed_cot.strip() + "\n\nThe final answer is: $\\boxed{" + verified_answer + "}$"


def compress_trace(compressor, text, ratio, *, family="qwen"):
    if not 0 < ratio <= 1:
        raise ValueError("Retention ratio must be in (0, 1]")
    if family not in ("qwen", "llama3"):
        raise ValueError("Unregistered upstream compression family")
    if ratio == 1:
        return {"compressed_prompt": text, "identity": True}
    kwargs = {"rate": ratio}
    if family == "llama3":
        kwargs.update(force_tokens=["Step", ":"], force_reserve_digit=True, drop_consecutive=True)
    return compressor.compress_prompt(text, **kwargs)


def dap_messages(question, full_trace, prompt_spec):
    """Appendix A framework plus explicit full-source input from section 3.1.

    The integration wrapper is an adaptation; the source text is never clipped.
    Difficulty assessment and the selected framework remain in the output.
    """
    return [
        {"role": "system", "content": prompt_spec["framework"]},
        {"role": "user", "content": prompt_spec["rewrite_wrapper"].format(
            question=question, source_trace=full_trace)},
    ]


def dap_structure(text):
    """Audit observable framework selection; do not invent an unreported label."""
    labels = re.findall(r"(?im)^\s*(?:\*\*)?Difficulty(?:\*\*)?\s*:\s*(easy|medium|hard)\b", text)
    return {
        "difficulty": labels[-1].lower() if labels else None,
        "has_analysis": bool(re.search(r"(?i)\banalysis\s*(?:\*\*)?\s*:", text)),
        "has_reflection": bool(re.search(r"(?i)\breflection\s*(?:\*\*)?\s*:", text)),
        "has_decomposition": "problem decomposition" in text.lower(),
        "has_solution": bool(re.search(r"(?i)\bsolution\s*(?:\*\*)?\s*:", text)),
    }
