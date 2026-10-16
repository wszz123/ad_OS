"""Audit both baseline and plan-review reports without model generation."""

import argparse
import json
from pathlib import Path

from coop.experiment_audit import audit_cost_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--output", default="results/cost_workload_audit.json")
    args = parser.parse_args()
    audits = [{"source_report": path, **audit_cost_report(json.loads(Path(path).read_text(encoding="utf-8")))}
              for path in args.report]
    Path(args.output).write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{key: value for key, value in audit.items() if key not in ("failures", "trial_metrics")}
                      for audit in audits], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
