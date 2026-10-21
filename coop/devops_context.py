"""Loss-aware projection of public evidence for model prompts; raw traces stay intact."""

from copy import deepcopy


def preview(value, *, depth=0):
    """Bound example values, never discard a check's name or pass/fail label."""
    if isinstance(value, str):
        return value if len(value) <= 1200 else value[:1200] + f" [truncated; {len(value)} characters]"
    if isinstance(value, list):
        if len(value) > 6:
            return {"count": len(value), "sample": [preview(item, depth=depth + 1) for item in value[:6]]}
        return [preview(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        if depth > 5:
            return {"fields": sorted(value)}
        return {key: preview(item, depth=depth + 1) for key, item in value.items()}
    return value


def suite_summary(suite):
    cases = []
    for case in suite.get("cases", []):
        checks = case.get("checks", [])
        cases.append({
            "scenario": case.get("scenario"), "variant": case.get("variant"),
            "passed": case.get("passed", False),
            "passed_checks": [check.get("name") for check in checks if check.get("passed")],
            "failures": [preview(check) for check in checks if not check.get("passed")],
            "error": case.get("error"), "runtime_errors": case.get("runtime_errors", 0),
            "operation_count": case.get("operation_count", 0),
        })
    return {"passed": suite.get("passed", False), "cases": cases,
            "runtime_errors": suite.get("runtime_errors", 0)}


def verification_summary(verification):
    result = {key: value for key, value in verification.items() if key != "checks"}
    result["checks"] = []
    for check in verification.get("checks", []):
        row = {key: value for key, value in check.items() if key != "suite"}
        if "suite" in check:
            row["suite"] = suite_summary(check["suite"])
        result["checks"].append(row)
    return result


def observations_summary(observations):
    result = []
    for observation in observations:
        row = {key: value for key, value in observation.items() if key not in ("suite", "inspection")}
        for key in ("error", "planner_error"):
            if key in row:
                row[key] = preview(row[key])
        if "suite" in observation:
            row["suite"] = suite_summary(observation["suite"])
        if "inspection" in observation:
            inspection = observation["inspection"]
            row["inspection"] = {key: value for key, value in inspection.items() if key != "recent_logs"}
            row["inspection"]["recent_logs"] = [
                {**{key: value for key, value in log.items() if key != "suite"},
                 "suite": suite_summary(log["suite"])} for log in inspection.get("recent_logs", [])]
        result.append(row)
    return result


def project_context(context):
    """Keep all current source, contracts, failures and mutation history available."""
    result = deepcopy(context)
    if "observations" in result:
        result["observations"] = observations_summary(result["observations"])
    if "verification" in result:
        result["verification"] = verification_summary(result["verification"])
    for memory in result.get("prior_experience", []):
        memory["observations"] = observations_summary(memory.get("observations", []))
    result["evidence_projection"] = "public_failures_v1; bounded examples, complete failure labels; raw evidence retained in task trace"
    return result
