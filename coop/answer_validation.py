"""Deterministic numeric, unit and citation grading; gold never enters the graph."""

from .calculator import decimal_value


def validate_cost_answer(task, answer, tool_result):
    def equal_number(value):
        try:
            return abs(decimal_value(value) - decimal_value(task.expected_value)) <= decimal_value("0.005")
        except (ValueError, TypeError):
            return False

    parsed = isinstance(answer, dict)
    numeric = parsed and equal_number(answer.get("value"))
    unit = parsed and answer.get("unit") == task.unit
    citations = answer.get("citations", []) if parsed else []
    valid_list = isinstance(citations, list) and all(isinstance(value, str) for value in citations)
    required = set(task.required_citations)
    actual = set(citations) if valid_list else set()
    tool_sources = set(tool_result.get("citations", []))
    grounded = valid_list and required.issubset(actual) and actual == tool_sources
    tool_correct = equal_number(tool_result.get("value")) and tool_result.get("unit") == task.unit
    return {"numeric_correct": bool(numeric), "unit_correct": bool(unit),
            "citations_correct": bool(grounded), "tool_numeric_correct": bool(tool_correct),
            "missing_citations": sorted(required - actual),
            "unbacked_citations": sorted(actual - tool_sources),
            "passed": bool(numeric and unit and grounded and tool_correct)}
