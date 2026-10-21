"""Re-score saved MoreAgent predictions against evaluator-only labels and write tables."""

import argparse
import json
from pathlib import Path

from tools.moreagent_data import load_questions, parse_answer, sha256, summarize


def verify_report(path, dataset=None):
    path = Path(path)
    report = json.loads(path.read_text())
    questions, manifest = load_questions(dataset or report["config"]["dataset"])
    if manifest["source_sha256"] != report["dataset"]["source_sha256"]:
        raise ValueError("source dataset changed since inference")
    selected = questions[:report["selected_questions"]]
    public = "\n".join(json.dumps(row["public"], ensure_ascii=False) for row in selected).encode()
    if (path.parent / "questions.public.jsonl").read_bytes() != public + b"\n":
        raise ValueError("public input artifact differs from whitelist extraction")
    if report.get("public_input_sha256") != sha256(public):
        raise ValueError("canonical public input digest mismatch")
    labels = [{"question_id": row["public"]["question_id"], "gold": row["gold"],
               "topic": row["topic"], "source_type": row["source_type"]} for row in selected]
    if json.loads((path.parent / "labels.evaluation-only.json").read_text()) != labels:
        raise ValueError("evaluator label artifact differs from supplied labels")
    expected = [row["public"]["question_id"] for row in selected]
    results = {}
    for method, cell in report["methods"].items():
        rows = cell["results"]
        if [row["question_id"] for row in rows] != expected[:len(rows)]:
            raise ValueError("duplicate, reordered or unexpected questions in results")
        if report["status"] == "complete" and len(rows) != len(selected):
            raise ValueError("complete report is missing predictions")
        for row, question in zip(rows, selected):
            if row["gold"] != question["gold"] or row["topic"] != question["topic"] or row["source_type"] != question["source_type"]:
                raise ValueError("evaluator labels or strata differ from source")
            prediction, parse_status = parse_answer(row.get("raw_prediction", ""))
            correct = row["status"] == "ok" and prediction == question["gold"]
            if correct != row.get("correct"):
                raise ValueError("saved correctness disagrees with raw prediction and supplied label")
            if row["status"] == "ok" and (prediction != row.get("prediction") or parse_status != row["parse_status"]):
                raise ValueError("saved parse disagrees with raw output")
        results[method] = summarize(rows)
    pairs = {}
    names = list(report["methods"])
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            left = {r["question_id"]: r for r in report["methods"][first]["results"]}
            right = {r["question_id"]: r for r in report["methods"][second]["results"]}
            cells = {"both_correct": [], "first_only": [], "second_only": [], "both_wrong": []}
            for question_id in expected:
                if question_id not in left or question_id not in right:
                    continue
                a, b = left[question_id]["correct"], right[question_id]["correct"]
                key = "both_correct" if a and b else "first_only" if a else "second_only" if b else "both_wrong"
                cells[key].append(question_id)
            pairs[first + "_vs_" + second] = {"first": first, "second": second, **cells}
    return report, {"source_report": str(path), "status": report["status"],
                    "requested_questions": len(selected), "methods": results, "paired": pairs,
                    "dataset_sha256_verified": True, "public_input_whitelist_verified": True,
                    "public_input_hash_policy": "canonical_jsonl_without_terminal_newline",
                    "raw_prediction_scores_verified": True}


def write_tables(report, summary, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "verified_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    lines = ["# MoreAgent 医学 50 题试测", "", f"- 运行状态：{report['status']}",
             f"- 模型：`{report['config']['model_path']}`", f"- 输出设置：`{report['config']['response_style']}`，"
             f"每次文本生成最多 {report['config']['max_new_tokens']} token，latent steps={report['config']['latent_steps']}。",
             "- 原始 30 题为开发使用过的 MedQA 题；20 题为项目自编，非官方未见测试集。",
             "- 无 RAG、无跨题记忆；答案与关联注释未进入 Agent 输入。",
             "- 文本上限相同但推理计算量不同；固定顺序单次运行、共享 GPU，不作为通用加速或显著性结论。", "",
             "| 方法 | 已尝试/请求 | 答对 | 准确率（已尝试） | 无效答案 | 推理秒 | 生成 token | 中间文本 token |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, cell in summary["methods"].items():
        accuracy = f"{cell['accuracy']:.1%}" if cell["accuracy"] is not None else "—"
        lines.append(f"| {name} | {cell['attempted']}/{summary['requested_questions']} | {cell['correct']} | {accuracy} | "
                     f"{cell['invalid_answers']} | {cell['elapsed_sec']:.2f} | {cell['generated_tokens']} | {cell['intermediate_text_tokens']} |")
    for field in ("source_type", "topic"):
        lines.extend(["", "## " + ("按来源" if field == "source_type" else "按主题"), "",
                      "| 分组 | 方法 | 答对/已尝试 | 准确率 |", "| --- | --- | ---: | ---: |"])
        for name, cell in summary["methods"].items():
            for group, value in cell["by_" + field].items():
                lines.append(f"| {group} | {name} | {value['correct']}/{value['count']} | {value['accuracy']:.1%} |")
    lines.extend(["", "## 逐题配对", ""])
    for pair in summary["paired"].values():
        lines.append(f"- {pair['first']} / {pair['second']}：都对 {len(pair['both_correct'])}；"
                     f"仅前者对 {len(pair['first_only'])}；仅后者对 {len(pair['second_only'])}；都错 {len(pair['both_wrong'])}。")
    lines.extend(["", "KV 为同进程引用交接，logical tensor bytes 不能直接当作网络传输字节。",
                  "原始输出、实际角色提示词及 KV 协议指标见逐题 trace；本文件根据原始预测重新评分。"])
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    args = parser.parse_args()
    report, summary = verify_report(args.report, args.dataset)
    output = args.output or str(Path(args.report).parent)
    write_tables(report, summary, output)
    print(json.dumps({"status": summary["status"], "output": output, "methods": summary["methods"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
