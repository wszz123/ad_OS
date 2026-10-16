"""Seeded parameter extrapolation and distractors; labels are evaluator-only."""

from dataclasses import replace
from decimal import Decimal, ROUND_HALF_UP
import json
import random

from .memory import MemoryItem
from .workload import cost_documents, cost_tasks


def reference_amount(template_index, inputs):
    """Independent financial oracle, not the runtime AST calculator.

    Original sources are unchanged in this experiment. Explicit decimal rates
    here belong only to grading; they never enter public_request or prompts.
    """
    values = {key: Decimal(value) for key, value in inputs.items()}
    if template_index < 5:
        capital = values["node_count"] * Decimal("2400") + Decimal("2000")
        electricity = (values["node_count"] * Decimal("0.080") * Decimal("10")
                       * values.get("days", Decimal(0)) * Decimal("0.75"))
        monthly = electricity + Decimal("300")
        amount = (capital if template_index == 0 else electricity if template_index == 1
                  else capital + monthly if template_index == 2
                  else capital + values["months"] * monthly)
    else:
        storage = (Decimal("120") * values.get("retention_days", Decimal(0))
                   * values.get("replicas", Decimal(0)) * Decimal("0.04"))
        egress = (Decimal("120") * values.get("days", Decimal(0))
                  * values.get("egress_fraction", Decimal(0)) * Decimal("0.09"))
        amount = (storage if template_index in (5, 9) else egress if template_index == 6
                  else storage + egress if template_index == 7
                  else values["months"] * (storage + egress))
    return str(amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def heldout_tasks(seed=20261010, rounds=1):
    """New parameters, existing task structures; not new-domain generalization."""
    if rounds < 1:
        raise ValueError("rounds must be positive")
    rng = random.Random(seed)
    result = []
    for block in range(rounds):
        nodes = rng.choice([3, 5, 7, 11, 13, 17])
        expanded_nodes = nodes + rng.choice([4, 6, 10])
        days = rng.choice([17, 21, 28, 31, 37, 43])
        months = rng.choice([3, 6, 9, 15, 18])
        retention = rng.choice([7, 14, 45, 60, 90])  # Intentionally different from billing days.
        replicas = rng.choice([3, 4, 5])
        changed_replicas = 3 + (replicas - 3 + 1) % 3
        fraction = rng.choice(["0.05", "0.15", "0.35", "0.40"])
        for index, template in enumerate(cost_tasks()):
            mapping = {"node_count": str(expanded_nodes if index == 4 else nodes), "days": str(days),
                       "months": str(months), "retention_days": str(retention),
                       "replicas": str(changed_replicas if index == 9 else replicas),
                       "egress_fraction": fraction}
            inputs = {key: mapping[key] for key in template.inputs}
            question = template.question.replace("first-year", "planning-period").replace("annual total", "planning-period total")
            result.append(replace(template, task_id=f"heldout-{block + 1:02d}-{template.task_id}",
                                  question=question, inputs=inputs,
                                  expected_value=reference_amount(index, inputs)))
    return result


def expanded_documents(distractors=128, seed=20261010):
    if distractors < 0:
        raise ValueError("distractors must be nonnegative")
    rng = random.Random(seed + 1)
    docs = cost_documents()
    for index in range(distractors):
        entity = f"Archive-{index:04d}"
        if index % 2:
            facts = {"daily_gb": str(rng.randint(10, 500)), "storage_rate": str(Decimal(rng.randint(1, 20)) / 100)}
            summary = f"{entity} archived storage quote with daily GB and replica costs. Not applicable to Lyra."
        else:
            facts = {"node_price": str(rng.randint(1000, 9000)), "power_watts": str(rng.randint(40, 300))}
            summary = f"{entity} archived node procurement and electricity quote. Not applicable to Orion."
        docs.append(MemoryItem(
            f"distractor-{index:04d}", "fixture", "2026-01-01T00:00:00+00:00", entity,
            summary + " Facts: " + ", ".join(f"{key}={value}" for key, value in facts.items()),
            memory_type="source", tags=["entity:" + entity, *("fact:" + key for key in facts)],
            evidence=json.dumps({"entity": entity, "facts": facts}, sort_keys=True),
        ))
    return docs
