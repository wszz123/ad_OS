"""Unit/sensitivity validation of generated arithmetic, without gold answers."""

import ast
from fractions import Fraction

from .calculator import expression_names


DIMENSIONS = {
    "node_count": {"node": 1}, "node_price": {"CNY": 1, "node": -1},
    "power_watts": {"energy": 1, "hour": -1, "node": -1},
    "hours_per_day": {"hour": 1, "day": -1}, "days": {"day": 1},
    "energy_rate": {"CNY": 1, "energy": -1}, "network_setup": {"CNY": 1},
    # Maintenance is the amount for one billing month; months is a scalar multiplier.
    "monthly_maintenance": {"CNY": 1}, "months": {},
    "daily_gb": {"GB": 1, "day": -1}, "retention_days": {"day": 1},
    "replicas": {}, "storage_rate": {"CNY": 1, "GB": -1},
    "egress_fraction": {}, "egress_rate": {"CNY": 1, "GB": -1},
}


def validate_plan_expression(expression, inputs):
    tree, names = expression_names(expression)
    missing = set(inputs) - names
    if missing:
        raise ValueError(f"expression ignores explicit user parameters: {sorted(missing)}")

    def units(node):
        if isinstance(node, ast.Constant):
            return {}
        if isinstance(node, ast.Name):
            if node.id not in DIMENSIONS:
                raise ValueError(f"no unit metadata for {node.id}")
            return DIMENSIONS[node.id]
        if isinstance(node, ast.UnaryOp):
            return units(node.operand)
        left, right = units(node.left), units(node.right)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            if left != right:
                raise ValueError(f"incompatible additive dimensions: {left} vs {right}")
            return left
        direction = -1 if isinstance(node.op, ast.Div) else 1
        result = dict(left)
        for dimension, exponent in right.items():
            result[dimension] = result.get(dimension, 0) + direction * exponent
        return {key: value for key, value in result.items() if value}

    actual = units(tree.body)
    if actual != {"CNY": 1}:
        raise ValueError(f"expected total currency dimensions, got {actual}")


def validate_quantity_contract(expression, question):
    """Check the fixture's public rate/quantity semantics symbolically.

    These are domain contracts, not task-specific gold formulas. No source
    values or expected totals are read. Polynomials expose additive offsets such
    as (1 + replicas) that dimensional analysis alone cannot detect.
    """
    tree, _ = expression_names(expression)

    def normalize(powers):
        return tuple(sorted((key, value) for key, value in powers.items() if value))

    def combine(left, right):
        result = {}
        for a, coefficient_a in left.items():
            for b, coefficient_b in right.items():
                powers = dict(a)
                for name, degree in b:
                    powers[name] = powers.get(name, 0) + degree
                key = normalize(powers)
                result[key] = result.get(key, Fraction(0)) + coefficient_a * coefficient_b
        if len(result) > 256:
            raise ValueError("symbolic expression is too complex")
        return {key: value for key, value in result.items() if value}

    def polynomial(node):
        if isinstance(node, ast.Constant):
            return {(): Fraction(str(node.value))}
        if isinstance(node, ast.Name):
            return {((node.id, 1),): Fraction(1)}
        if isinstance(node, ast.UnaryOp):
            multiplier = -1 if isinstance(node.op, ast.USub) else 1
            return {key: multiplier * value for key, value in polynomial(node.operand).items()}
        left, right = polynomial(node.left), polynomial(node.right)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            result = dict(left)
            sign = -1 if isinstance(node.op, ast.Sub) else 1
            for key, value in right.items():
                result[key] = result.get(key, Fraction(0)) + sign * value
            return {key: value for key, value in result.items() if value}
        if isinstance(node.op, ast.Div):
            if len(right) != 1:
                raise ValueError("quantity checker requires a nonzero monomial divisor")
            (powers, coefficient), = right.items()
            if not coefficient:
                raise ValueError("division by zero")
            right = {tuple((name, -degree) for name, degree in powers): 1 / coefficient}
        return combine(left, right)

    terms = polynomial(tree.body)
    question = question.lower()
    for monomial in terms:
        powers = dict(monomial)
        if powers.get("storage_rate"):
            for name in ("daily_gb", "retention_days", "replicas"):
                if powers.get(name) != 1:
                    raise ValueError(f"each storage cost term must be linear in {name}; replicas is total copies, not additional copies")
        if powers.get("egress_rate"):
            if powers.get("replicas", 0):
                raise ValueError("egress charges original data, not replicated copies")
            for name in ("daily_gb", "days", "egress_fraction"):
                if powers.get(name) != 1:
                    raise ValueError(f"each egress cost term must be linear in {name}")
        if powers.get("power_watts") and powers.get("node_count") != 1:
            raise ValueError("each per-node electricity cost term must scale with node_count")
        if "exclude recurring" in question and any(powers.get(name) for name in ("monthly_maintenance", "energy_rate")):
            raise ValueError("the user explicitly excludes recurring costs")
        if "exclude egress" in question and powers.get("egress_rate"):
            raise ValueError("the user explicitly excludes egress cost")
