"""Re-evaluate saved task snapshots without rerunning or prompting the model."""

import argparse
import json
from pathlib import Path
import tempfile

from coop.devops_world import DevOpsWorld, digest, tasks


def evaluate_saved(report_path):
    path = Path(report_path)
    report = json.loads(path.read_text())
    if report.get("workload") == "constructed_catalog_project_v2":
        from coop.catalog_world import CatalogProjectWorld
        factory = CatalogProjectWorld
        sequence = factory.task_factory()
        extension = ".json"
    elif report.get("workload") == "constructed_catalog_service_v1":
        factory, sequence, extension = DevOpsWorld, tasks(), ".py"
    else:
        raise ValueError("unsupported workload")
    rows = report.get("tasks", [])
    requested = report.get("requested_tasks", 10)
    if type(requested) is not int or not 1 <= requested <= 10:
        raise ValueError("invalid requested task count")
    sequence = sequence[:requested]
    if [r["task_id"] for r in rows] != [t["task_id"] for t in sequence]:
        raise ValueError("report must contain the complete requested ordered prefix")
    result = {"source_report": str(path), "evaluation": "saved_snapshots_current_evaluator",
              "workload": report["workload"], "requested_tasks": requested,
              "model_rerun": False, "tasks": []}
    dependencies_pass = True
    with tempfile.TemporaryDirectory() as directory:
        world = factory(directory)
        for task, row in zip(sequence, rows):
            item = {"task_id": task["task_id"], "original_status": row["status"],
                    "dependencies_pass": dependencies_pass, "passed": False}
            if row["status"] == "blocked":
                item["reason"] = "original execution was blocked"
            else:
                try:
                    if hasattr(world, "prepare_task"):
                        world.prepare_task(task)
                    candidate = (path.parent / (task["task_id"] + "-candidate" + extension)).read_text()
                    deployed = (path.parent / (task["task_id"] + "-deployed" + extension)).read_text()
                    target = candidate if task["group"] == "dev" else deployed
                    if digest(target) != row.get("grade", {}).get("version"):
                        raise ValueError("snapshot does not match recorded evaluation version")
                    world.source.write_text(candidate)
                    world.deployed.write_text(deployed)
                    grade = world.evaluate(task, row.get("observations", []))
                    item["grade"] = grade
                    item["passed"] = dependencies_pass and row["status"] == "verified" and grade["passed"]
                except Exception as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
            dependencies_pass = dependencies_pass and item["passed"]
            result["tasks"].append(item)
    result["passed_tasks"] = sum(row["passed"] for row in result["tasks"])
    result["status"] = ("complete" if requested == 10 else "partial") if result["passed_tasks"] == requested else "failed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.report).resolve() == Path(args.output).resolve():
        parser.error("output must differ from the original report")
    result = evaluate_saved(args.report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"status": result["status"], "passed_tasks": result["passed_tasks"], "output": str(output)}))
    if result["status"] not in ("complete", "partial"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
