"""Trusted subprocess harness for the allowlisted multi-module service exercise."""

import builtins
import json
import os
import resource
import sys
import types


ORDER = ("repository", "policies", "limits", "cache", "pagination", "mutations", "gateway", "service")


def run(payload):
    modules = {}

    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in modules:
            raise ImportError("only previously loaded project modules may be imported")
        return modules[name]

    safe = {name: getattr(builtins, name) for name in (
        "str", "int", "bool", "min", "max", "len", "list", "dict", "tuple", "sorted", "isinstance")}
    safe["__import__"] = local_import
    for name in ORDER:
        module = types.ModuleType(name)
        module.__dict__["__builtins__"] = safe
        exec(compile(payload["files"][name + ".py"], name + ".py", "exec"), module.__dict__)
        modules[name] = module
    state, records = payload["state"], payload["records"]
    defaults, overrides = payload["defaults"], payload["overrides"]
    outputs, logs = [], []
    for index, operation in enumerate(payload["operations"]):
        kind = operation["type"]
        before_hits, before_revision = state["cache_hits"], state["revision"]
        try:
            if kind == "search":
                request = dict(operation["request"])
                cursor_from = operation.get("cursor_from")
                if cursor_from is not None:
                    request["cursor"] = outputs[cursor_from].get("next_cursor")
                output = modules["service"].search(records, request, state, defaults,
                                                    overrides.get(request["tenant"], {}))
            elif kind == "update":
                records, output = modules["gateway"].apply(records, operation["command"], state)
            elif kind == "configure":
                defaults.update(operation.get("defaults", {}))
                for tenant, values in operation.get("overrides", {}).items():
                    overrides.setdefault(tenant, {}).update(values)
                output = {"configured": True}
            else:
                raise ValueError("unknown operation")
        except Exception as exc:
            output = {"runtime_error": f"{type(exc).__name__}: {exc}"}
        outputs.append(output)
        logs.append({"step": index, "operation": kind, "revision": state["revision"],
                     "tenant": operation.get("request", operation.get("command", {})).get("tenant"),
                     "revision_before": before_revision, "cache_hit": state["cache_hits"] > before_hits,
                     "cache_hits": state["cache_hits"], "cache_misses": state["cache_misses"],
                     "error": output.get("error", output.get("runtime_error"))})
    return {"outputs": outputs, "state": state, "records": records, "defaults": defaults,
            "overrides": overrides, "logs": logs, "worker_pid": os.getpid()}


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    print(json.dumps(run(json.load(sys.stdin))))
