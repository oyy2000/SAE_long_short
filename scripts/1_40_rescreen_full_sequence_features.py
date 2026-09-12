#!/usr/bin/env python3
"""Check local prerequisites or re-screen sealed SAE scores under the v2 rule."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from length_budget_distill.experiment_io import read_json, write_json_exclusive
from length_budget_distill.sae_full_sequence_screen import preflight,rescreen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','screen'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--source-root',help='Relocated parent artifact root; existing registered hashes still must match.')
    parser.add_argument('--report',help='Exclusive JSON preflight report path.')
    args=parser.parse_args()
    config=read_json(args.config)
    if args.action=='screen':
        result=rescreen(config,args.source_root)
        print(json.dumps({'status':result['status'],'stable_features':result['stable_features']},indent=2))
    else:
        result=preflight(config,args.source_root)
        if args.report:
            Path(args.report).parent.mkdir(parents=True,exist_ok=True)
            write_json_exclusive(args.report,result)
        print(json.dumps(result,indent=2))
        if result['status']!='ready':sys.exit(2)


if __name__=='__main__':main()
