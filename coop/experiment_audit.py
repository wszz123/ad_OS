"""Derived comparisons and provenance checks over completed experiment reports."""

from statistics import mean


def reduction(baseline, current):
    return 100 * (1 - current / baseline) if baseline else None


def audit_report(report):
    trials = []
    checks = []
    for index, trial in enumerate(report["trials"]):
        modes = trial["modes"]
        metrics = {}
        for mode, data in modes.items():
            rows = data["tasks"]
            by_task = {row["task_id"]: row for row in rows}
            positions = {row["task_id"]: position for position, row in enumerate(rows)}
            if len(by_task) != len(rows):
                raise ValueError("duplicate task id in a trial")
            accepts = [row for row in rows if row.get("accepted_reuse")]
            for row in accepts:
                source = row.get("reuse_source_task")
                valid = (source in by_task and positions[source] < positions[row["task_id"]]
                         and by_task[source]["status"] == "ok"
                         and row.get("reuse_source_agent") == "summarizer"
                         and by_task[source].get("evidence_id") == row.get("evidence_id")
                         and by_task[source].get("evidence") == row.get("evidence"))
                checks.append({"trial": index, "mode": mode, "task_id": row["task_id"],
                               "source_task_id": source, "provenance_valid": valid,
                               "evidence_matches_label": row["evidence_correct"]})
            metrics[mode] = {
                "completed": data["completed"], "evidence_accuracy_proxy": data["evidence_accuracy_proxy"],
                "model_calls": data["model_calls"], "prompt_tokens": data["prompt_tokens"],
                "generated_tokens": data["generated_tokens"], "message_tokens": data["message_tokens"],
                "total_transfer_bytes": data["total_transfer_bytes"], "elapsed_sec": data["elapsed_sec"],
                "accepted_reuse": len(accepts),
                "reuse_evidence_label_accuracy": sum(row["evidence_correct"] for row in accepts) / len(accepts) if accepts else None,
                "new_memory_records": data["new_memory_records"],
            }
        if "text" in modes and "protocol" in modes:
            text, protocol = modes["text"], modes["protocol"]
            inputs_equal = (text["prompt_tokens"] == protocol["prompt_tokens"]
                            and text["generated_tokens"] == protocol["generated_tokens"])
            outputs_equal = [(row["task_id"], row.get("evidence_id"), row.get("answer")) for row in text["tasks"]] == [
                (row["task_id"], row.get("evidence_id"), row.get("answer")) for row in protocol["tasks"]]
            metrics["protocol"].update(
                token_totals_equal_to_text=inputs_equal, answers_equal_to_text=outputs_equal,
                communication_token_reduction_vs_text_pct=reduction(text["message_tokens"], protocol["message_tokens"]),
                transfer_byte_reduction_vs_text_pct=reduction(text["total_transfer_bytes"], protocol["total_transfer_bytes"]),
            )
        if "protocol_vector" in modes and "protocol_vector_memory" in modes:
            baseline, memory = modes["protocol_vector"], modes["protocol_vector_memory"]
            metrics["protocol_vector_memory"].update(
                model_calls_saved=baseline["model_calls"] - memory["model_calls"],
                corpus_searches_saved=baseline["corpus_searches"] - memory["corpus_searches"],
                prompt_token_reduction_vs_vector_pct=reduction(baseline["prompt_tokens"], memory["prompt_tokens"]),
                generated_token_reduction_vs_vector_pct=reduction(baseline["generated_tokens"], memory["generated_tokens"]),
                task_time_reduction_vs_vector_pct=reduction(baseline["elapsed_sec"], memory["elapsed_sec"]),
            )
        trials.append(metrics)
    averages = {}
    for mode in {mode for trial in trials for mode in trial}:
        rows = [trial[mode] for trial in trials if mode in trial]
        averages[mode] = {key: mean(row[key] for row in rows if row.get(key) is not None)
                          for key in rows[0] if any(row.get(key) is not None for row in rows)
                          and not isinstance(rows[0].get(key), bool)}
    return {"experiment": report["experiment"], "trial_count": len(trials),
            "trial_metrics": trials, "averages": averages, "reuse_checks": checks,
            "all_reuse_provenance_valid": all(check["provenance_valid"] for check in checks),
            "scope": "Evidence-label proxy and observed local warm timings, not answer correctness or general speedup"}


def audit_cost_report(report):
    """Compare quality and communication separately for the sourced workload."""
    trials = []
    failures = []
    for index, trial in enumerate(report["trials"]):
        data = trial["modes"]
        metrics = {}
        for mode, row in data.items():
            metrics[mode] = {key: row.get(key, 0) for key in (
                "completed", "passed_count", "quality_pass_rate", "model_calls", "plan_reviews",
                "plan_repairs", "plan_attempt_count", "plan_repair_exhausted",
                "message_tokens", "total_transfer_bytes", "prompt_tokens", "generated_tokens",
                "accepted_reuse", "corpus_searches", "source_parses", "cache_validation_parses", "elapsed_sec",
            )}
            for task in row["tasks"]:
                if not task["quality"]["passed"]:
                    failures.append({"trial": index, "mode": mode, "task_id": task["task_id"],
                                     "status": task["status"], "error": task.get("error"),
                                     "expression": task.get("expression"), "expected_value": task["expected_value"],
                                     "actual_value": task.get("answer", {}).get("value"), "quality": task["quality"]})
        if "text" in data and "protocol" in data:
            text, protocol = data["text"], data["protocol"]
            metrics["protocol"]["communication_token_reduction_vs_text_pct"] = reduction(text["message_tokens"], protocol["message_tokens"])
            metrics["protocol"]["transfer_byte_reduction_vs_text_pct"] = reduction(text["total_transfer_bytes"], protocol["total_transfer_bytes"])
            metrics["protocol"]["answers_equal_to_text"] = [task.get("answer") for task in text["tasks"]] == [
                task.get("answer") for task in protocol["tasks"]]
        if "protocol_vector" in data and "protocol_vector_memory" in data:
            baseline, reused = data["protocol_vector"], data["protocol_vector_memory"]
            metrics["protocol_vector_memory"]["corpus_searches_saved"] = baseline["corpus_searches"] - reused["corpus_searches"]
            metrics["protocol_vector_memory"]["model_calls_saved"] = baseline["model_calls"] - reused["model_calls"]
        trials.append(metrics)
    averages = {}
    for mode in sorted({mode for trial in trials for mode in trial}):
        rows = [trial[mode] for trial in trials if mode in trial]
        averages[mode] = {key: mean(row[key] for row in rows) for key in rows[0]
                          if not isinstance(rows[0][key], bool)}
    return {"experiment": report["experiment"], "review_plan": report["config"].get("review_plan", False),
            "repair_plan": report["config"].get("repair_plan", False),
            "max_plan_repairs": report["config"].get("max_plan_repairs") if report["config"].get("repair_plan") else None,
            "device": report["config"]["device"], "embedding_device": report["config"]["embedding_device"],
            "trial_count": len(trials), "averages": averages, "trial_metrics": trials, "failures": failures,
            "scope": "Numeric/unit/citation checks on synthetic fixtures; explanation prose not graded"}
