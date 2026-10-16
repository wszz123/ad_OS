"""Audit retained real-model reports without rerunning the models."""

import argparse
import json
from pathlib import Path

from coop.experiment_audit import audit_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="results/qwen_continuous_v2.json")
    parser.add_argument("--output", default="results/qwen_continuous_v2_audit.json")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    audit = audit_report(report)
    Path(args.output).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"trial_count": audit["trial_count"],
                      "averages": audit["averages"],
                      "all_reuse_provenance_valid": audit["all_reuse_provenance_valid"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
