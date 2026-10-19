"""Stdlib diagnostics and explicitly scripted, non-LLM demonstration support."""

import importlib.metadata
import json
from pathlib import Path
import re
import platform
import socket
import sys


QWEN = "/home/Users/SharedFile/Models/Qwen/Qwen3-8B"
EMBEDDING = "/home/Users/SharedFile/Models/Qwen/Qwen3-Embedding-4B"


def environment_report(*, backend="fixture", model_path=QWEN, embedding_path=EMBEDDING):
    required = ["torch", "numpy", "langgraph", "langgraph-checkpoint", "langgraph-checkpoint-sqlite"]
    if backend == "qwen":
        required.append("transformers")
    if backend == "qwen":
        required.extend(["safetensors", "datasets", "accelerate"])
    packages = {}
    for name in dict.fromkeys(required + ["transformers"]):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    missing = [name for name in required if packages[name] is None]
    models = {}
    for name, location in (("generation", model_path), ("embedding", embedding_path)):
        path = Path(location)
        missing_files = [filename for filename in ("config.json", "tokenizer_config.json") if not (path / filename).is_file()]
        weights = list(path.glob("*.safetensors")) if path.is_dir() else []
        if not weights:
            missing_files.append("*.safetensors")
        index = path / "model.safetensors.index.json"
        if index.is_file():
            try:
                references = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
                missing_files.extend(filename for filename in references if not (path / filename).is_file())
            except (ValueError, KeyError, TypeError):
                missing_files.append("valid model.safetensors.index.json")
        models[name] = {"path": str(path), "ready": not missing_files, "missing_files": sorted(set(missing_files))}
    proxy_open = False
    try:
        with socket.create_connection(("127.0.0.1", 1081), timeout=0.5):
            proxy_open = True
    except OSError:
        pass
    gpu = {"available": False, "devices": []}
    if packages["torch"]:
        try:
            import torch
            gpu["available"] = torch.cuda.is_available()
            if gpu["available"]:
                for index in range(torch.cuda.device_count()):
                    free, total = torch.cuda.mem_get_info(index)
                    gpu["devices"].append({"index": index, "name": torch.cuda.get_device_name(index),
                                           "free_bytes": free, "total_bytes": total})
        except Exception as exc:
            gpu["error"] = f"{type(exc).__name__}: {exc}"
    python_ok = sys.version_info[:2] == (3, 10)
    required_models = ["generation", "embedding"] if backend == "qwen" else []
    return {"experiment": "prototype_environment_diagnostic", "backend": backend,
            "python": platform.python_version(), "python_3_10": python_ok,
            "platform": platform.system(), "packages": packages, "missing_packages": missing,
            "models": models, "gpu": gpu, "proxy_127_0_0_1_1081_open": proxy_open,
            "ready": python_ok and not missing and all(models[name]["ready"] for name in required_models),
            "scope": "No model weights loaded. Proxy port reachability is not protocol/authentication validation."}


class WordCounter:
    def encode(self, text, **kwargs):
        return text.split()


class FixturePlanner:
    """Domain recipes from public request intent, never task IDs/gold values.

    Only for wiring demos. This is not model inference, reasoning evaluation or
    Qwen token measurement. Arithmetic still comes from sourced runtime facts.
    """
    model_path = "fixture:public-intent-recipes-v1"

    def __init__(self):
        self.tokenizer = WordCounter()
        self.call_stats = []

    def generate_json(self, messages, **kwargs):
        content = messages[-1]["content"]
        if "Tool result: " in content:
            tool = json.loads(content.split("Tool result: ", 1)[1])
            value = {key: tool[key] for key in ("value", "unit", "citations")}
            value["explanation"] = "Scripted summary of the sourced calculator result."
        elif "expression" in messages[0]["content"]:
            # Public-intent recipe; intentionally not a language model or task-ID lookup.
            input_match = re.search(r"User inputs: (\{[^\n]*\})", content)
            entity_match = re.search(r"Entity: ([^\n]+)", content)
            task_match = re.search(r"Task: ([^\n]+)", content)
            if not (input_match and entity_match and task_match):
                raise ValueError("fixture planner could not parse the public task fields")
            inputs = json.loads(input_match.group(1))
            entity, question = entity_match.group(1), task_match.group(1).lower()
            capital = "node_count * node_price + network_setup"
            electricity = "node_count * power_watts / 1000 * hours_per_day * days * energy_rate"
            storage = "daily_gb * retention_days * replicas * storage_rate"
            egress = "daily_gb * days * egress_fraction * egress_rate"
            if entity == "Orion":
                expression = (capital if "exclude recurring" in question else electricity if "electricity cost" in question
                              else f"{capital} + months * ({electricity} + monthly_maintenance)"
                              if "first-year" in question or "planning-period" in question
                              else f"{capital} + {electricity} + monthly_maintenance")
            elif entity == "Lyra":
                expression = (f"months * ({storage} + {egress})" if "annual total" in question or "planning-period" in question
                              else f"{storage} + {egress}" if "monthly total" in question
                              else storage if "storage cost" in question else egress)
            else:
                raise ValueError("fixture demo only handles the declared Orion/Lyra schema")
            catalog_line = next((line.split(": ", 1)[1] for line in content.splitlines()
                                 if line.startswith("Source variable meanings")), "{}")
            allowed_names = set(json.loads(catalog_line)) | set(inputs)
            if not set(re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*", expression)) <= allowed_names:
                raise ValueError("fixture plan references a variable outside the public schema")
            value = {"search_query": entity + " cost facts", "expression": expression}
        else:
            raise ValueError("unsupported fixture planner prompt")
        self.call_stats.append({"prompt_tokens": sum(len(message["content"].split()) for message in messages),
                                "generated_tokens": len(json.dumps(value).split()), "elapsed_sec": 0.0})
        return {"value": value}


def fixture_encode(texts, *, query=False):
    """Two-dimensional topic fixture, not a learned semantic embedding."""
    return [[1.0, 0.0] if text.startswith("Orion") else [0.0, 1.0] for text in texts]
