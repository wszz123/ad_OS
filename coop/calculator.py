"""Bounded arithmetic over sourced numeric facts, without Python eval/exec."""

import ast
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from typing import Dict


def decimal_value(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("value must be a finite number or decimal string")
    if len(str(value)) > 64:
        raise ValueError("numeric value is too long")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid decimal value") from exc
    if not number.is_finite() or abs(number) > Decimal("1e12"):
        raise ValueError("numeric value outside finite range")
    return number


def expression_names(expression: str):
    if not isinstance(expression, str) or not expression or len(expression) > 512:
        raise ValueError("expression must contain 1 to 512 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid arithmetic expression") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > 128:
        raise ValueError("expression is too complex")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UAdd, ast.USub)
    if any(not isinstance(node, allowed) for node in nodes):
        raise ValueError("only numbers, variable names and + - * / are allowed")
    for node in nodes:
        if isinstance(node, ast.Constant):
            decimal_value(node.value)
    return tree, {node.id for node in nodes if isinstance(node, ast.Name)}


def calculate(expression: str, inputs: Dict, facts: Dict, unit="CNY"):
    """Facts map variable names to {value, source_id}; inputs are explicit user data."""
    tree, required = expression_names(expression)
    if set(inputs) & set(facts):
        raise ValueError("user inputs must not override sourced facts")
    values = {name: decimal_value(value) for name, value in inputs.items()}
    values.update({name: decimal_value(record["value"]) for name, record in facts.items()})
    missing = required - set(values)
    if missing:
        raise ValueError(f"unknown arithmetic variables: {sorted(missing)}")

    def evaluate(node):
        if isinstance(node, ast.Constant):
            return decimal_value(ast.get_source_segment(expression, node))
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        left, right = evaluate(node.left), evaluate(node.right)
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        elif isinstance(node.op, ast.Mult):
            value = left * right
        else:
            if not right:
                raise ValueError("division by zero")
            value = left / right
        return decimal_value(value)

    with localcontext() as context:
        context.prec = 32
        value = evaluate(tree.body)
        rounded = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    citations = sorted({facts[name]["source_id"] for name in required if name in facts})
    return {"value": str(rounded), "unit": unit, "expression": expression,
            "used_variables": sorted(required), "citations": citations,
            "source_values": {name: {"value": str(values[name]),
                                     "source_id": facts[name]["source_id"] if name in facts else "user"}
                              for name in sorted(required)}}
