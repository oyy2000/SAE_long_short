"""Dataset loading utilities."""

from __future__ import annotations

from typing import Any, Dict, List

from .config import resolve_path
from .records import ProblemRecord, read_jsonl
from .verifiers import extract_final_answer


def load_problem_records(config: Dict[str, Any]) -> List[ProblemRecord]:
    dataset = dict(config.get("dataset", {}))
    source = dataset.get("source", "local_jsonl")
    if source == "local_jsonl":
        path = resolve_path(dataset.get("path"), config)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        raw = list(read_jsonl(path))
    elif source == "hf_dataset":
        from datasets import load_dataset

        loaded = load_dataset(
            dataset["dataset_name"],
            dataset.get("dataset_config"),
            split=dataset.get("split", "train"),
        )
        raw = list(loaded)
    else:
        raise ValueError(f"Unsupported dataset source: {source}")
    limit = dataset.get("max_examples")
    if limit is not None:
        raw = raw[: int(limit)]
    question_field = dataset.get("question_field", "question")
    answer_field = dataset.get("answer_field", "answer")
    records = []
    for index, item in enumerate(raw):
        answer = str(item[answer_field])
        if dataset.get("answer_format") in {"gsm8k", "gsm8k_final"}:
            extracted = extract_final_answer(answer)
            if extracted is None:
                raise ValueError(f"Could not extract GSM8K answer at row {index}")
            answer = extracted
        records.append(
            ProblemRecord(
                problem_id=str(item.get("id", f"hf-{index:06d}")),
                question=str(item[question_field]),
                answer=answer,
                metadata={
                    key: value
                    for key, value in item.items()
                    if key not in {"id", question_field, answer_field}
                },
            )
        )
    return records
