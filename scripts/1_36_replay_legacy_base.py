#!/usr/bin/env python3
"""Replay the frozen base evaluation on another GPU architecture, without replacing evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from length_budget_distill.legacy_replication import config_load, source_hashes, root_for, json_lines, marker_write
from length_budget_distill.experiment_io import publish_files_hash_verified, write_json_exclusive
from length_budget_distill.factorial import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = config_load(args.config)
    source_hashes(config)
    root = root_for(config)
    runtime = Path(tempfile.mkdtemp(prefix='youyang7-legacy-base-check-', dir='/var/tmp'))
    command = [sys.executable, str(root / 'legacy_code/scripts/4_1_eval_model.py'), '--config', str(root / 'replication/eval_config.json'), '--model-name', 'Qwen/Qwen2.5-1.5B-Instruct', '--start-index', '0', '--limit', str(config['evaluation']['count']), '--output-jsonl', str(runtime / 'predictions.jsonl'), '--summary-json', str(runtime / 'summary.json'), '--max-new-tokens', str(config['evaluation']['max_new_tokens']), '--batch-size', str(config['evaluation']['batch_size']), '--temperature', '0', '--top-p', '1', '--torch-dtype', 'bfloat16']
    subprocess.run(command, cwd=root / 'legacy_code', check=True)
    rows = json_lines(runtime / 'predictions.jsonl')
    comparisons = {}
    for label, filename in [('historical', root / 'imported/historical_base_predictions.jsonl'), ('replication_c31', root / 'replication/evaluation/base/predictions.jsonl')]:
        other = json_lines(filename)
        if len(rows) != len(other) or any((a['problem_id'], a['question'], a['gold_answer']) != (b['problem_id'], b['question'], b['gold_answer']) for a,b in zip(rows, other)):
            raise ValueError('Base replay question/gold mismatch.')
        comparisons[label] = {field: sum(a[field] != b[field] for a,b in zip(rows,other)) for field in ('prediction_text', 'predicted_answer', 'is_correct', 'output_token_count')}
    report = {'node': os.uname().nodename, 'n': len(rows), 'accuracy': sum(r['is_correct'] for r in rows)/len(rows), 'comparisons': comparisons, 'changes_primary_gate_baseline': False}
    write_json_exclusive(runtime / 'comparison.json', report)
    output = root / 'replication/base_replay_c49'
    publish_files_hash_verified(runtime, output, ('predictions.jsonl', 'summary.json', 'comparison.json'))
    marker_write(output / 'BASE_REPLAY_COMPLETE', {'status': 'complete', 'comparison_sha256': file_sha256(output / 'comparison.json')})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
