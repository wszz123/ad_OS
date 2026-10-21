"""Independent scorer and integrity checks for a native medical MAS report."""

import argparse
import json
from pathlib import Path

from tools.moreagent_data import load_questions, parse_answer, sha256, summarize
from coop.medical_memory import exact_quotes


def verify_report(path, dataset=None):
    path = Path(path)
    report = json.loads(path.read_text())
    if report.get("implementation") != "child repository only; no parent methods/models imports":
        raise ValueError("report is not from the native child-repository graph")
    questions, manifest = load_questions(dataset or report["config"]["dataset"])
    if report["dataset"]["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("presentation source digest changed")
    selected = questions[:report["selected_questions"]]
    if len(selected) != report["selected_questions"]:
        raise ValueError("scheduled question count mismatch")
    public = "\n".join(json.dumps(row["public"], ensure_ascii=False) for row in selected).encode()
    if report["public_input_sha256"] != sha256(public):
        raise ValueError("public input digest mismatch")
    directory = path.parent
    if (directory / "questions.public.jsonl").read_bytes() != public + b"\n":
        raise ValueError("public question artifact differs from whitelist conversion")
    labels = [{"question_id": row["public"]["question_id"], "gold": row["gold"],
               "topic": row["topic"], "source_type": row["source_type"]} for row in selected]
    if json.loads((directory / "labels.evaluation-only.json").read_text()) != labels:
        raise ValueError("independent evaluator labels do not match source dataset")
    result = {"experiment": "native_medical_report_independent_verification", "source_report": str(path),
              "model_rerun": False, "requested": len(selected), "methods": {}, "paired": {}, "skipped_modes": {}}
    fully_executed = True
    for mode, cell in report["modes"].items():
        rows = cell["results"]
        if [row["sequence"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"{mode}: duplicate or reordered rows")
        if cell.get("status") not in ("complete", "complete_with_errors"):
            result["skipped_modes"][mode] = {"status": cell.get("status"), "rows": len(rows)}
            fully_executed = False
            continue
        if len(rows) != len(selected):
            raise ValueError(f"{mode}: completed report missing rows")
        written, first_source = set(), {}
        rescored = []
        link_audit = {"declared_memory_refs": 0, "cross_topic_refs": 0,
                      "matched_candidate_annotation_refs": 0, "by_source": {}}
        for row, source in zip(rows, selected):
            if row["question_id"] != source["public"]["question_id"]:
                raise ValueError(f"{mode}: question order mismatch")
            if (row["gold"], row["topic"], row["source_type"]) != (source["gold"], source["topic"], source["source_type"]):
                raise ValueError(f"{mode}: evaluator labels or strata changed")
            if row["status"] == "ok":
                final = row.get("final")
                if not isinstance(final, dict) or final.get("answer") not in "ABCD":
                    raise ValueError(f"{mode}/{row['question_id']}: invalid final output")
                try:
                    raw_final = json.loads(row["raw_prediction"])
                except (json.JSONDecodeError, TypeError):
                    raise ValueError(f"{mode}/{row['question_id']}: raw JSON final response is invalid")
                if final["answer"] != row["prediction"] or raw_final != final:
                    raise ValueError(f"{mode}/{row['question_id']}: answer and raw output mismatch")
                correct = row["prediction"] == source["gold"]
                if row["correct"] != correct:
                    raise ValueError(f"{mode}/{row['question_id']}: score mismatch (correctness mismatch)")
                memory_ids = [item["memory_id"] for item in row["retrieval"]["memories"]]
                if any(item not in written for item in memory_ids):
                    raise ValueError(f"{mode}/{row['question_id']}: self/future/uncommitted memory was supplied")
                selected_notes = {item["memory_id"]: item for item in row["retrieval"]["memories"]}
                if len(final.get("used_memory_ids", [])) != len(set(final.get("used_memory_ids", []))):
                    raise ValueError(f"{mode}/{row['question_id']}: duplicate declared memory use")
                for used in final["used_memory_ids"]:
                    item = selected_notes.get(used)
                    if item is None or not item["source_quotes"]:
                        raise ValueError(f"{mode}/{row['question_id']}: used memory has no source quote")
                    source_question = selected[item["source_sequence"] - 1]["public"]["question"] if 1 <= item["source_sequence"] <= len(selected) else None
                    if first_source.get(used) != (item["source_sequence"], item["source_question"]) or source_question != item["source_question"]:
                        raise ValueError(f"{mode}/{row['question_id']}: source quotation provenance changed")
                    expected_quotes = exact_quotes(source_question, [quote["text"] for quote in item["source_quotes"]])
                    if expected_quotes != item["source_quotes"]:
                        raise ValueError(f"{mode}/{row['question_id']}: source quotes do not match their question")
                    if item["source_sequence"] >= row["sequence"] or row["question_id"] in item["source_question"]:
                        raise ValueError(f"{mode}/{row['question_id']}: non-prior memory source")
                used_checks = {check["memory_id"]: check for check in final.get("condition_checks", [])}
                if set(final["used_memory_ids"]) - set(used_checks):
                    raise ValueError(f"{mode}/{row['question_id']}: used memory lacks a condition recheck")
                if any(used_checks[memory_id].get("decision") != "used"
                       or not used_checks[memory_id].get("current_conditions") for memory_id in final["used_memory_ids"]):
                    raise ValueError(f"{mode}/{row['question_id']}: missing current-condition reasoning")
                related = set(source.get("evaluator_related_prior_ids", []))
                for used in final["used_memory_ids"]:
                    item = selected_notes[used]
                    source_id = item["source_question_id"]
                    link_audit["declared_memory_refs"] += 1
                    cross_topic = item["topic"] != source.get("topic")
                    link_audit["cross_topic_refs"] += int(cross_topic)
                    matched = source_id in related
                    link_audit["matched_candidate_annotation_refs"] += int(matched)
                    per_source = link_audit["by_source"].setdefault(source_id, {"uses": 0, "cross_topic_uses": 0})
                    per_source["uses"] += 1
                    per_source["cross_topic_uses"] += int(cross_topic)
                if row["memory_writes"]:
                    admission = row["admission"]
                    if (admission.get("clinical_status") != "unverified_model_note"
                            or not admission.get("written")
                            or type(admission.get("exact_quotes")) is not int or admission["exact_quotes"] < 1):
                        raise ValueError("medical note was mislabeled as clinically verified")
                    if not final.get("memory_note"):
                        raise ValueError("memory written without a model note")
                    written.add(admission["memory_id"])
                    first_source[admission["memory_id"]] = (row["sequence"], source["public"]["question"])
                elif row["admission"].get("written"):
                    raise ValueError("admission flag and memory write counter disagree")
            row_copy = dict(row)
            row_copy["correct"] = row.get("status") == "ok" and row.get("prediction") == source["gold"]
            rescored.append(row_copy)
        metrics = summarize(rescored)
        for key in ("model_calls", "budget_saturated_calls", "message_count", "message_chars", "message_bytes",
                    "state_transfers", "state_payload_bytes", "state_metadata_bytes", "total_transfer_bytes",
                    "memory_queries", "memory_candidate_count", "retrieval_hits", "declared_cross_topic_reuse",
                    "correct_with_declared_reuse", "schema_retries", "role_calls", "embedding_sec"):
            metrics[key] = sum(row.get(key, 0) for row in rows)
        metrics["declared_reuse"] = sum(len(row.get("final", {}).get("used_memory_ids", [])) for row in rows)
        metrics["actual_memory_writes"] = len(written)
        metrics["retrieval_hit_rate"] = metrics["retrieval_hits"] / metrics["memory_queries"] if metrics["memory_queries"] else None
        metrics["offline_candidate_link_audit"] = link_audit
        result["methods"][mode] = metrics
        result["methods"][mode]["correct_with_declared_reuse"] = sum(
            row.get("correct", False) and bool(row.get("final", {}).get("used_memory_ids")) for row in rows)
    modes = list(result["methods"])
    for index, left_mode in enumerate(modes):
        for right_mode in modes[index + 1:]:
            left = {row["question_id"]: row for row in report["modes"][left_mode]["results"]}
            right = {row["question_id"]: row for row in report["modes"][right_mode]["results"]}
            pair = {key: [] for key in ("both_correct", "left_only", "right_only", "both_wrong")}
            for source in selected:
                qid = source["public"]["question_id"]
                if qid not in left or qid not in right:
                    continue
                a = left[qid].get("prediction") == source["gold"]
                b = right[qid].get("prediction") == source["gold"]
                key = "both_correct" if a and b else "left_only" if a else "right_only" if b else "both_wrong"
                pair[key].append(qid)
            result["paired"][left_mode + "_vs_" + right_mode] = pair
    profile_keys = ("backend", "model_path", "embedding_path", "top_k", "schema_attempts", "seed", "max_samples")
    result["comparison_profile"] = {key: report["config"].get(key) for key in profile_keys}
    result["device_profile"] = {"generation_device": report["config"].get("device"),
                                 "embedding_device": report["config"].get("embedding_device")}
    result["selected_questions"] = report["selected_questions"]
    result["source_sha256"] = report["dataset"]["source_sha256"]
    result["status"] = "complete" if fully_executed else "partial"
    return result


def combine_verified(summaries):
    methods, by_mode = {}, {}
    comparison_profile, source_hash, selected_count = None, None, None
    for summary in summaries:
        if comparison_profile is None:
            comparison_profile = summary["comparison_profile"]
            source_hash, selected_count = summary["source_sha256"], summary["selected_questions"]
        elif (summary["comparison_profile"] != comparison_profile or summary["source_sha256"] != source_hash
              or summary["selected_questions"] != selected_count):
            raise ValueError("paired medical reports must share model/run configuration, source digest and question count")
        raw = json.loads(Path(summary["source_report"]).read_text())
        for mode, values in summary["methods"].items():
            if mode in methods:
                raise ValueError(f"duplicate completed method: {mode}")
            methods[mode] = values
            by_mode[mode] = {row["question_id"]: row for row in raw["modes"][mode]["results"]}
    paired = {}
    names = list(methods)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            left, right = by_mode[left_name], by_mode[right_name]
            if set(left) != set(right):
                continue
            cells = {key: [] for key in ("both_correct", "left_only", "right_only", "both_wrong")}
            for qid in left:
                a, b = left[qid]["correct"], right[qid]["correct"]
                key = "both_correct" if a and b else "left_only" if a else "right_only" if b else "both_wrong"
                cells[key].append(qid)
            paired[left_name + "_vs_" + right_name] = cells
    statuses = [summary["status"] for summary in summaries]
    expected_methods = {"text", "protocol", "protocol_vector", "protocol_vector_group_memory", "protocol_vector_memory"}
    all_modes_complete = (set(methods) == expected_methods and selected_count is not None
                          and all(value["attempted"] == selected_count for value in methods.values()))
    return {"experiment": "native_medical_paired_independent_verification", "methods": methods,
            "paired": paired, "source_reports": [summary["source_report"] for summary in summaries],
            "device_profiles": {str(index): summary["device_profile"] for index, summary in enumerate(summaries)},
            "status": "complete" if all_modes_complete else "partial",
            "source_sha256": source_hash, "selected_questions": selected_count,
            "comparison_profile": comparison_profile,
            "note": "Methods from separate runs are paired by identical question IDs and labels. Device assignment differs; latency is not directly paired and shared-GPU contention may differ."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="+")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    args = parser.parse_args()
    verified_reports = [verify_report(path, args.dataset) for path in args.report]
    verified = combine_verified(verified_reports) if len(verified_reports) > 1 else verified_reports[0]
    output = Path(args.output or Path(args.report[0]).parent / "verified_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verified, ensure_ascii=False, indent=2))
    lines = ["# 子仓库原生医学 MAS 独立重评分", "",
             f"- 状态：`{verified['status']}`；试题数：{verified.get('selected_questions', verified.get('requested'))}",
             "- 答案由数据文件独立复核；无效格式与运行错误按错题计入准确率分母。", "",
             "| 模式 | 答对/题数 | 准确率 | 运行错误 | 声明记忆使用 | 写入笔记 | 推理秒 | 生成 token |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, cell in verified["methods"].items():
        lines.append(f"| {name} | {cell['correct']}/{cell['attempted']} | {cell['accuracy']:.1%} | {cell['errors']} | "
                     f"{cell.get('declared_reuse', 0)} | {cell.get('actual_memory_writes', 0)} | "
                     f"{cell['elapsed_sec']:.1f} | {cell['generated_tokens']} |")
    for name, pair in verified["paired"].items():
        lines.append(f"- {name}: both={len(pair['both_correct'])}, first-only={len(pair['left_only'])}, "
                     f"second-only={len(pair['right_only'])}, neither={len(pair['both_wrong'])}.")
    lines.extend(["", "Source of truth is `verified_summary.json`; raw report and traces retain prompt/runtime evidence.",
                  "Medical memory notes are explicitly unverified model outputs. Declared reuse is not clinical evidence validation."])
    output.with_name("summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": verified["status"], "output": str(output),
                      "methods": verified["methods"], "paired": verified["paired"]}, ensure_ascii=False, indent=2))
    if any(summary["status"] == "failed" for summary in verified_reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
