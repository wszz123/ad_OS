"""Synthetic, multi-source cost-analysis tasks with evaluator-only gold answers."""

from dataclasses import dataclass
import json
from typing import Dict, List, Tuple

from .memory import MemoryItem


@dataclass(frozen=True)
class CostTask:
    task_id: str
    entity: str
    question: str
    inputs: Dict[str, str]
    expected_value: str
    required_citations: Tuple[str, ...]
    unit: str = "CNY"


def variable_catalog(entity):
    """Public tool/schema descriptions, without source values or task formulas."""
    catalogs = {
        "Orion": {
            "node_price": "CNY per purchased node; procurement cost only, not electricity",
            "power_watts": "watts per operating node",
            "hours_per_day": "operating hours per node per day",
            "energy_rate": "CNY per kWh of electricity",
            "network_setup": "CNY one-time for the entire deployment, not per node",
            "monthly_maintenance": "CNY per month for the entire deployment, not per node",
        },
        "Lyra": {
            "daily_gb": "GB of original new data per day",
            "storage_rate": "CNY per stored GB per month; stored volume includes replicas",
            "egress_rate": "CNY per retrieved original GB; retrieval does not include replica copies",
        },
    }
    if entity not in catalogs:
        raise ValueError(f"unknown entity schema: {entity}")
    return catalogs[entity]


def cost_documents() -> List[MemoryItem]:
    records = [
        ("orion-price", "Orion", "Orion deployment node procurement price.", {"node_price": "2400"}),
        ("orion-power", "Orion", "Orion deployment node power and daily operation hours.", {"power_watts": "80", "hours_per_day": "10"}),
        ("orion-tariff", "Orion", "Orion deployment electricity tariff in CNY per kWh.", {"energy_rate": "0.75"}),
        ("orion-ops", "Orion", "Orion deployment network setup capital cost and monthly maintenance in CNY.", {"network_setup": "2000", "monthly_maintenance": "300"}),
        ("lyra-workload", "Lyra", "Lyra archive produces daily original data measured in GB.", {"daily_gb": "120"}),
        ("lyra-storage", "Lyra", "Lyra archive monthly storage tariff in CNY per GB including every replica.", {"storage_rate": "0.04"}),
        ("lyra-egress", "Lyra", "Lyra archive retrieval traffic tariff in CNY per original GB, not multiplied by replicas.", {"egress_rate": "0.09"}),
        ("other-price", "Vega", "Vega legacy node procurement price; not applicable to Orion.", {"node_price": "3900"}),
        ("other-storage", "Nova", "Nova archive storage tariff; not applicable to Lyra.", {"storage_rate": "0.08"}),
    ]
    return [MemoryItem(
        doc_id, "fixture", "2026-01-01T00:00:00+00:00", entity,
        description + " Facts: " + ", ".join(f"{key}={value}" for key, value in facts.items()),
        memory_type="source", tags=[f"entity:{entity}", *(f"fact:{key}" for key in facts)],
        evidence=json.dumps({"entity": entity, "facts": facts}, sort_keys=True),
    ) for doc_id, entity, description, facts in records]


def cost_tasks() -> List[CostTask]:
    capital = ("orion-price", "orion-ops")
    energy = ("orion-power", "orion-tariff")
    all_orion = capital + energy
    storage = ("lyra-workload", "lyra-storage")
    traffic = ("lyra-workload", "lyra-egress")
    all_lyra = ("lyra-workload", "lyra-storage", "lyra-egress")
    return [
        CostTask("orion-01", "Orion", "Calculate deployment capital cost: buy the nodes and pay the one-time network setup. Exclude recurring costs.",
                 {"node_count": "8"}, "21200.00", capital),
        CostTask("orion-02", "Orion", "Calculate monthly electricity cost for the nodes over the stated days. Convert watts to kW; use documented daily operating hours.",
                 {"node_count": "8", "days": "30"}, "144.00", energy),
        CostTask("orion-03", "Orion", "Calculate first-month total: node procurement, one-time network setup, monthly electricity and monthly maintenance.",
                 {"node_count": "8", "days": "30"}, "21644.00", all_orion),
        CostTask("orion-04", "Orion", "Calculate first-year total: capital cost once plus electricity and maintenance each month. Each month has the stated days.",
                 {"node_count": "8", "days": "30", "months": "12"}, "26528.00", all_orion),
        CostTask("orion-05", "Orion", "Recalculate first-year total for the expanded node count: capital cost once plus electricity and maintenance each month. Network setup and monthly maintenance stay fixed.",
                 {"node_count": "12", "days": "30", "months": "12"}, "36992.00", all_orion),
        CostTask("lyra-01", "Lyra", "Calculate monthly storage cost: daily original GB times retention days times replicas times the monthly storage tariff.",
                 {"retention_days": "30", "replicas": "2"}, "288.00", storage),
        CostTask("lyra-02", "Lyra", "Calculate monthly egress cost: the specified fraction of original GB produced during the month is retrieved. Do not multiply traffic by replicas.",
                 {"days": "30", "egress_fraction": "0.2"}, "64.80", traffic),
        CostTask("lyra-03", "Lyra", "Calculate monthly total: replicated storage at the stated retention plus retrieval traffic for the month's original data fraction.",
                 {"retention_days": "30", "replicas": "2", "days": "30", "egress_fraction": "0.2"}, "352.80", all_lyra),
        CostTask("lyra-04", "Lyra", "Calculate annual total: multiply the monthly replicated storage and original-data retrieval traffic costs by months. Storage retention is fixed each month.",
                 {"retention_days": "30", "replicas": "2", "days": "30", "egress_fraction": "0.2", "months": "12"}, "4233.60", all_lyra),
        CostTask("lyra-05", "Lyra", "Calculate monthly storage cost after changing replicas to the provided count. Exclude egress cost.",
                 {"retention_days": "30", "replicas": "1"}, "144.00", storage),
    ]
