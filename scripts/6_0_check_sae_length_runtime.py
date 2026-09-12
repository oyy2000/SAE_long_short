"""Check C31 runtime and original feature assets without generating responses."""
import argparse
import json
from pathlib import Path

from length_budget_distill.sae_runtime_preflight import runtime_preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = runtime_preflight(args.config, args.project_root, args.output)
    print(json.dumps({'status': report['status'], 'blockers': report['blockers'],
                      'report': str(args.output / 'preflight.json')}, ensure_ascii=False), flush=True)
    raise SystemExit(2 if report['blockers'] else 0)


if __name__ == '__main__':
    main()
