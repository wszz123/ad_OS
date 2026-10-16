"""Compact evidence summaries for the competition prototype only."""

import json
from pathlib import Path
from statistics import mean


def summarize_report(data, source):
    kind = data.get("experiment")
    row = {"source": str(source), "experiment": kind}
    if kind == "four_role_sourced_cost_workload":
        row["evidence"] = "scripted_fixture" if data.get("demo_backend") == "fixture" else "real_model_synthetic_tasks"
        row["runtime"] = data.get("runtime", {})
        modes = {}
        for name in sorted({name for trial in data.get("trials", []) for name in trial["modes"]}):
            values = [trial["modes"][name] for trial in data["trials"] if name in trial["modes"]]
            modes[name] = {key: mean(value.get(key, 0) for value in values) for key in (
                "task_count", "completed", "passed_count", "model_calls", "prompt_tokens", "generated_tokens",
                "message_tokens", "total_transfer_bytes", "accepted_reuse", "corpus_searches",
                "state_transfers", "state_payload_bytes")}
        row["metrics"] = modes
        row["all_quality_passed"] = bool(modes) and all(
            value["passed_count"] == value["task_count"] for value in modes.values())
        row["scope"] = "Per-trial means; repeated tasks are not independent unique benchmark samples."
    elif kind == "unified_evidence_summary":
        return None
    elif kind in ("planner_node_replay_with_atomic_repair_reservations", "sqlite_checkpoint_state_sigkill_recovery"):
        fake = data.get("fake", data.get("config", {}).get("fake", False))
        row.update(evidence="real_process_fixture_models" if fake else "real_model_recovery",
                   all_checks_passed=data.get("all_passed", False), scope=data.get("scope"))
        row["metrics"] = ([{"case": item["case"], "resume_calls": item["resume"]["calls"],
                            "status": item["resume"]["status"], "checks": item["checks"]}
                           for item in data["cases"]] if "cases" in data else data.get("checks", {}))
    elif kind in ("paired_memory_backend_comparison", "sqlite_storage_pushdown_projection_dedup_ablation"):
        row.update(evidence="real_model_synthetic_tasks", all_checks_passed=data.get("all_checks_passed", False),
                   metrics=data.get("summary", {}), scope=data.get("limitations", []))
    elif kind == "sqlite_atomic_batch_ingestion":
        row.update(evidence="cached_vector_storage_and_optional_real_workflow",
                   all_checks_passed=data.get("all_checks_passed", False),
                   metrics=data.get("ingestion_summary", {}), scope=data.get("limitations", []))
    elif kind == "frozen_policy_parameter_holdout_and_corpus_scaling":
        row.update(evidence="real_model_known_structure_parameter_holdout",
                   metrics=[{"corpus_size": cell["corpus_size"], **cell.get("summary", {})} for cell in data["cells"]],
                   scope=data.get("limitations", []))
    else:
        return None
    return row


def collect_reports(paths):
    evidence, skipped = [], []
    for path in sorted(set(Path(path) for path in paths)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            summary = summarize_report(data, path) if isinstance(data, dict) else None
            if summary is None:
                skipped.append({"source": str(path), "reason": "unsupported/development/summary schema"})
            else:
                evidence.append(summary)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            skipped.append({"source": str(path), "reason": f"{type(exc).__name__}: {exc}"})
    return {"experiment": "unified_evidence_summary", "reports": evidence, "skipped": skipped,
            "scope": "No heterogeneous grand pass-rate or speedup is calculated; fixture evidence is labelled explicitly."}


def write_summary(summary, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 原型实验摘要", "", "按报告列出证据，不混合计算总准确率或通用加速比。", ""]
    for row in summary["reports"]:
        lines.extend([f"## {row['experiment']}", f"- 来源：`{row['source']}`", f"- 证据类型：`{row['evidence']}`"])
        for key in ("all_quality_passed", "all_checks_passed"):
            if key in row:
                lines.append(f"- {key}: {row[key]}")
        lines.extend(["", "```json", json.dumps(row.get("metrics", {}), ensure_ascii=False, indent=2), "```", ""])
    lines.append(f"未纳入报告数：{len(summary['skipped'])}，原因见 summary.json。")
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
