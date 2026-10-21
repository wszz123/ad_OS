"""Multi-file, stateful catalog project with reproducible forked incident sessions."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import symtable
import tempfile

from .catalog_project import FILES, INCIDENTS, PATCHES, project_tasks
from .devops_graph import ScriptedDevOps
from .devops_world import digest


FUNCTIONS = {
    "repository": ("select", ["records", "tenant"]),
    "policies": ("resolve", ["defaults", "overrides"]),
    "limits": ("bound_limit", ["value", "policy"]),
    "cache": ("cache_key", ["request", "revision", "policy"]),
    "pagination": ("page", ["rows", "request", "revision"]),
    "mutations": ("commit", ["records", "command", "revision"]),
    "gateway": ("apply", ["records", "command", "state"]),
    "service": ("search", ["records", "request", "state", "defaults", "overrides"]),
}


def seed_records(hidden=False):
    tenants = ("orange", "purple", "green") if hidden else ("red", "blue", "green")
    base = 2000 if hidden else 0
    rows = []
    # Interleave tenants and reverse IDs: isolation and stable tie-breaking matter.
    for i in range(120, 0, -1):
        for tenant_index, tenant in enumerate(tenants):
            name = "Shared Widget" if i <= (12 if hidden else 16) else f"Widget {i:03}"
            if i % 29 == 0:
                name = f"Straße {i:03}"
            rows.append({"tenant": tenant, "id": base + i, "name": name,
                         "active": i % (7 if hidden else 5) != 0,
                         "stock": 100 + tenant_index * 300 + i + (37 if hidden else 0)})
    return rows


class BundleView:
    def __init__(self, world, target):
        self.world, self.target = world, target

    def read_text(self, encoding="utf-8"):
        return self.world.manifest(self.target).read_text(encoding=encoding)

    def write_text(self, text, encoding="utf-8"):
        self.world.write_bundle(self.target, json.loads(text))


class CatalogProjectWorld:
    project_id = "catalog-project-v2"
    workload_id = "constructed_catalog_project_v2"
    snapshot_extension = ".json"
    requires_inspection = True
    task_factory = staticmethod(project_tasks)
    guide = ("Multi-file catalog project: service.py -> repository/policies/limits/cache/pagination; "
             "gateway.py -> mutations.py. Read docs/architecture.md and docs/policy.md, not obsolete v0 rules. "
             "Inspect SQLite/config/log evidence; patch a list of {path,old,new} atomically. "
             "Each probe runs a reproducible forked multi-request session; code and evidence persist across tasks.")

    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.source_reads = self.tool_calls = self.patch_files = self.probe_operations = 0
        self.stage = -1
        self.source, self.deployed = BundleView(self, "source"), BundleView(self, "deployed")
        for target in ("source", "deployed"):
            if not self.manifest(target).exists():
                self.write_bundle(target, FILES)
        with sqlite3.connect(self.root / "catalog.sqlite") as db:
            db.execute("CREATE TABLE IF NOT EXISTS records (tenant TEXT,id INTEGER,name TEXT,active INTEGER,stock INTEGER,PRIMARY KEY(tenant,id))")
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value INTEGER)")
            if not db.execute("SELECT COUNT(*) FROM records").fetchone()[0]:
                db.executemany("INSERT INTO records VALUES (:tenant,:id,:name,:active,:stock)", seed_records())
                db.executemany("INSERT INTO metadata VALUES (?,?)", [("data_revision", 1), ("policy_revision", 1)])
        if not (self.root / "control-plane.json").exists():
            (self.root / "control-plane.json").write_text(json.dumps({"include_archived": False, "max_limit": 20}))

    def manifest(self, target):
        if target not in ("source", "deployed"):
            raise ValueError("unknown project target")
        return self.root / (target + ".bundle.json")

    def bundle(self, target="source"):
        return json.loads(self.manifest(target).read_text())

    @staticmethod
    def validate_files(files):
        if set(files) != set(FILES) or any(not isinstance(text, str) for text in files.values()):
            raise ValueError("project file inventory must remain fixed")
        allowed = (ast.Module, ast.ImportFrom, ast.alias, ast.FunctionDef, ast.arguments, ast.arg,
                   ast.Assign, ast.Return, ast.Name, ast.Load, ast.Store, ast.Constant, ast.Call,
                   ast.Attribute, ast.ListComp, ast.comprehension, ast.Subscript, ast.Slice,
                   ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.UnaryOp, ast.USub, ast.Not,
                   ast.Compare, ast.In, ast.NotIn, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
                   ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.BoolOp, ast.And, ast.Or,
                   ast.List, ast.Tuple, ast.Dict, ast.IfExp, ast.If, ast.For, ast.Lambda,
                   ast.Expr, ast.keyword)
        builtins = {"str", "int", "bool", "min", "max", "len", "list", "dict", "tuple", "sorted", "isinstance"}
        methods = {"get", "casefold", "strip", "lower", "items", "update", "copy", "append", "setdefault", "pop"}
        for module, (function, arguments) in FUNCTIONS.items():
            text = files[module + ".py"]
            if len(text) > 16000:
                raise ValueError("module exceeds source budget")
            tree = ast.parse(text)
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            if len(functions) != 1 or any(not isinstance(n, (ast.ImportFrom, ast.FunctionDef)) for n in tree.body):
                raise ValueError("each module permits imports and one contract function only")
            fn = functions[0]
            if (fn.name != function or [a.arg for a in fn.args.args] != arguments or fn.decorator_list
                    or fn.args.defaults or fn.args.kw_defaults or fn.args.kwonlyargs
                    or fn.args.vararg or fn.args.kwarg or fn.args.posonlyargs or fn.returns
                    or any(a.annotation for a in fn.args.args)):
                raise ValueError("module contract signature changed")
            calls = builtins | {function}
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    if node.level or node.module not in FUNCTIONS:
                        raise ValueError("only project imports are allowed")
                    for alias in node.names:
                        if alias.name != FUNCTIONS[node.module][0] or (alias.asname or "").startswith("_"):
                            raise ValueError("unknown project import")
                        calls.add(alias.asname or alias.name)
            for node in ast.walk(tree):
                if not isinstance(node, allowed):
                    raise ValueError(f"unsupported project syntax: {type(node).__name__}")
                if isinstance(node, ast.Name) and node.id.startswith("_"):
                    raise ValueError("private names are not allowed")
                if isinstance(node, ast.Attribute) and node.attr not in methods:
                    raise ValueError("unsupported method")
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id not in calls:
                        raise ValueError("unsupported callable")
                    if not isinstance(node.func, (ast.Name, ast.Attribute)):
                        raise ValueError("unsupported call form")
            symbols = symtable.symtable(text, module + ".py", "exec")
            pending = list(symbols.get_children())
            while pending:
                scope = pending.pop()
                unresolved = sorted(symbol.get_name() for symbol in scope.get_symbols()
                                    if symbol.is_referenced() and symbol.is_global()
                                    and symbol.get_name() not in calls)
                if unresolved:
                    raise ValueError(f"{module}.py: undefined globals {unresolved}; "
                                     f"function parameters are {arguments}")
                pending.extend(scope.get_children())

    def write_bundle(self, target, files):
        self.validate_files(files)
        text = json.dumps(files, ensure_ascii=False, sort_keys=True, indent=2)
        release = self.root / "releases" / digest(text)
        for name, content in files.items():
            path = release / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        # The single authoritative manifest switches all modules together.
        temporary = self.manifest(target).with_suffix(".tmp")
        temporary.write_text(text)
        temporary.replace(self.manifest(target))

    def version(self, target="source"):
        return digest(self.manifest(target).read_text())

    def read_source(self):
        self.source_reads += 1
        return self.source.read_text()

    def memory_context(self):
        with sqlite3.connect(self.root / "catalog.sqlite") as db:
            return dict(db.execute("SELECT key,value FROM metadata"))

    def prepare_task(self, task):
        if task["index"] == self.stage:
            return
        self.stage = task["index"]
        with sqlite3.connect(self.root / "catalog.sqlite") as db:
            if self.stage == 2:
                db.execute("INSERT OR IGNORE INTO records VALUES ('red',9001,'Shared Widget',1,321)")
                db.execute("UPDATE metadata SET value=2 WHERE key='data_revision'")
            if self.stage == 3:
                db.execute("UPDATE metadata SET value=2 WHERE key='policy_revision'")
                (self.root / "control-plane.json").write_text(json.dumps({"include_archived": True, "max_limit": 7}))

    def _records(self, hidden):
        if hidden:
            return seed_records(True)
        with sqlite3.connect(self.root / "catalog.sqlite") as db:
            db.row_factory = sqlite3.Row
            return [{**dict(row), "active": bool(row["active"])} for row in db.execute("SELECT * FROM records ORDER BY rowid")]

    def _session(self, target, operations, records, checkpoint=None):
        state = {"revision": self.memory_context()["data_revision"], "cache": {}, "receipts": {},
                 "cache_hits": 0, "cache_misses": 0}
        payload = {"files": self.bundle(target), "operations": operations, "records": records, "state": state,
                   "defaults": {"include_archived": False, "max_limit": 20}, "overrides": {}}
        if checkpoint:
            payload.update({key: checkpoint[key] for key in ("records", "state", "defaults", "overrides")})
        process = subprocess.run([sys.executable, "-I", str(Path(__file__).with_name("catalog_worker.py"))],
                                 input=json.dumps(payload), text=True, capture_output=True, timeout=5, cwd=self.root)
        if process.returncode:
            raise ValueError("project worker failed: " + process.stderr[-1500:])
        return json.loads(process.stdout)

    @staticmethod
    def _expected(records, tenant, query="shared", archived=False, limit=3):
        rows = [r for r in records if r["tenant"] == tenant and query.strip().casefold() in r["name"].casefold()
                and (archived or r["active"])]
        rows = sorted(rows, key=lambda r: (r["name"].casefold(), r["id"]))[:limit]
        return [{key: row[key] for key in ("tenant", "id", "name", "stock")} for row in rows]

    def scenario(self, target, incident, *, hidden=False, variant=0):
        records = self._records(hidden)
        tenants = sorted({r["tenant"] for r in records})
        tenant, other = tenants[variant % len(tenants)], tenants[(variant + 1) % len(tenants)]
        query = "shared" if not variant else "widget"
        checks = []
        operations = []
        prior_outputs = []
        worker_pids = []

        def search(who=tenant, **values):
            request = {"tenant": who, "query": query, "limit": 3, **values}
            return {"type": "search", "request": request}

        def equal(name, actual, expected):
            checks.append({"name": name, "passed": actual == expected, "actual": actual, "expected": expected})

        def items(output):
            return output.get("items")

        if incident == "tenant_cache":
            operations = [search(), search(other), search()]
            result = self._session(target, operations, records)
            a, b, repeated = result["outputs"]
            def isolated(output, who):
                eligible = self._expected(records, who, query, limit=len(records))
                values = items(output)
                return isinstance(values, list) and len(values) == 3 and all(value in eligible for value in values)
            equal("tenant_isolation", isolated(a, tenant), True)
            equal("cache_namespace", isolated(b, other), True)
            equal("same_tenant_replay", items(repeated), items(a))
            equal("cache_is_used", result["state"]["cache_hits"] >= 1, True)
        elif incident == "cursor":
            operations = [search(limit=2), {**search(limit=2), "cursor_from": 0},
                          {**search(other, limit=2), "cursor_from": 0},
                          {**search(query="Straße", limit=2), "cursor_from": 0}, search(limit=3)]
            result = self._session(target, operations, records)
            a, b, foreign, changed, wider = result["outputs"]
            expected = self._expected(records, tenant, query, limit=4)
            equal("first_page", items(a), expected[:2])
            equal("duplicate_name_continuation", items(b), expected[2:])
            equal("foreign_cursor_rejected", foreign.get("error"), "invalid_cursor")
            equal("query_cursor_rejected", changed.get("error"), "invalid_cursor")
            equal("pagination_cache_key", items(wider), expected[:3])
        elif incident == "invalidation":
            first = self._expected(records, tenant, query)[0]
            operations = [search(), {"type": "update", "command": {
                "tenant": tenant, "idempotency_key": "stock-update", "changes": [{"id": first["id"], "delta": -7}]}},
                search(), {**search(), "cursor_from": 0}]
            result = self._session(target, operations, records)
            before, update, after, stale = result["outputs"]
            expected = self._expected(records, tenant, query)
            expected[0]["stock"] -= 7
            equal("update_succeeded", update.get("ok"), True)
            equal("read_after_write", items(after), expected)
            equal("revision_advanced_once", after.get("revision"), before.get("revision", -2) + 1)
            equal("stale_cursor_rejected", stale.get("error"), "invalid_cursor")
        elif incident == "policy":
            operations = [
                {"type": "configure", "defaults": {"include_archived": True, "max_limit": 7},
                 "overrides": {tenant: {"include_archived": False, "max_limit": 7}}},
                search(limit=7), {"type": "configure", "overrides": {tenant: {"include_archived": True}}},
                search(limit=7), {"type": "configure", "overrides": {tenant: {"include_archived": "false", "max_limit": 3}}},
                search(limit=20), search(limit=0)]
            result = self._session(target, operations, records)
            outputs = result["outputs"]
            equal("explicit_false_override", items(outputs[1]), self._expected(records, tenant, query, limit=7))
            equal("policy_cache_invalidation", items(outputs[3]), self._expected(records, tenant, query, archived=True, limit=7))
            equal("string_false_and_tenant_cap", items(outputs[5]), self._expected(records, tenant, query, limit=3))
            equal("zero_limit", items(outputs[6]), [])
        elif incident == "transactions":
            selected = self._expected(records, tenant, query)
            first, second = selected[:2]
            other_first = self._expected(records, other, query)[0]
            initial_revision = self.memory_context()["data_revision"]
            invalid = {"id": 999999, "delta": -1} if variant == 0 else {"id": second["id"], "delta": -100000}
            command = {"tenant": tenant, "idempotency_key": "release-batch", "expected_revision": initial_revision,
                       "changes": [{"id": first["id"], "delta": -3}, {"id": second["id"], "delta": -4}]}
            operations = [{"type": "update", "command": {"tenant": tenant, "idempotency_key": "invalid-batch",
                           "changes": [{"id": first["id"], "delta": -5}, invalid]}},
                          {"type": "update", "command": command}]
            phase1 = self._session(target, operations, records)
            prior_outputs = phase1["outputs"]
            expected_error = "not_found" if variant == 0 else "out_of_stock"
            equal("invalid_batch_rejected", phase1["outputs"][0].get("error"), expected_error)
            phase2_ops = [{"type": "update", "command": command},
                          {"type": "update", "command": {**command, "changes": [{"id": first["id"], "delta": -9}]}},
                          {"type": "update", "command": {"tenant": other, "idempotency_key": "release-batch",
                           "changes": [{"id": other_first["id"], "delta": -3}]}},
                          {"type": "update", "command": {"tenant": tenant, "idempotency_key": "stale-revision",
                           "expected_revision": initial_revision, "changes": [{"id": first["id"], "delta": -2}]}}, search()]
            # A new child process receives the serialized checkpoint: receipts must survive restart.
            with tempfile.TemporaryDirectory(prefix="checkpoint-", dir=self.root) as checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir) / "session.sqlite"
                with sqlite3.connect(checkpoint_path) as db:
                    db.execute("CREATE TABLE snapshot (payload TEXT NOT NULL)")
                    db.execute("INSERT INTO snapshot VALUES (?)", (json.dumps(phase1),))
                with sqlite3.connect(checkpoint_path) as db:
                    restored = json.loads(db.execute("SELECT payload FROM snapshot").fetchone()[0])
                result = self._session(target, phase2_ops, records, checkpoint=restored)
            worker_pids = [phase1["worker_pid"], result["worker_pid"]]
            equal("new_worker_restored_checkpoint", worker_pids[0] != worker_pids[1], True)
            stocks = {(r["tenant"], r["id"]): r["stock"] for r in result["records"]}
            equal("atomic_and_replay_first_stock", stocks[(tenant, first["id"])], first["stock"] - 3)
            equal("atomic_and_replay_second_stock", stocks[(tenant, second["id"])], second["stock"] - 4)
            equal("conflicting_replay_rejected", result["outputs"][1].get("error"), "idempotency_conflict")
            equal("tenant_scoped_receipt", stocks[(other, other_first["id"])], other_first["stock"] - 3)
            equal("optimistic_conflict", result["outputs"][3].get("error"), "revision_conflict")
            equal("no_duplicate_revision", result["state"]["revision"], initial_revision + 2)
            operations += phase2_ops
            result["logs"] = [{"phase": 1, **row} for row in phase1["logs"]] + [{"phase": 2, **row} for row in result["logs"]]
        else:
            raise ValueError("unknown incident scenario")
        errors = sum(bool(output.get("runtime_error")) for output in prior_outputs + result["outputs"])
        return {"scenario": incident, "hidden": hidden, "variant": variant,
                "passed": all(check["passed"] for check in checks) and not errors,
                "checks": checks, "operation_count": len(operations), "worker_restarts": int(incident == "transactions"),
                "worker_pids": worker_pids or [result["worker_pid"]],
                "cache_hits": result["state"]["cache_hits"], "runtime_errors": errors, "logs": result["logs"]}

    def _suite(self, target, incident, hidden=False):
        cases = []
        for i in range(2):
            try:
                cases.append(self.scenario(target, incident, hidden=hidden, variant=i))
            except Exception as exc:
                cases.append({"passed": False, "checks": [], "operation_count": 0,
                              "runtime_errors": 1, "error": f"{type(exc).__name__}: {exc}"})
        return {"passed": all(case["passed"] for case in cases), "cases": cases,
                "operation_count": sum(case["operation_count"] for case in cases),
                "runtime_errors": sum(case["runtime_errors"] for case in cases)}

    def execute(self, task, action):
        self.tool_calls += 1
        tool = action.get("tool")
        if tool == "inspect":
            files = self.bundle()
            with sqlite3.connect(self.root / "catalog.sqlite") as db:
                counts = dict(db.execute("SELECT tenant,COUNT(*) FROM records GROUP BY tenant"))
            log_path = self.root / "observations.jsonl"
            logs = [json.loads(line) for line in log_path.read_text().splitlines()[-2:]] if log_path.exists() else []
            return {"inspection": {"file_lines": {name: len(text.splitlines()) for name, text in files.items()},
                                   "module_contracts": {name + ".py": {"function": fn, "parameters": params}
                                                        for name, (fn, params) in FUNCTIONS.items()},
                                   "dependencies": {name: [node.module for node in ast.parse(text).body
                                                           if isinstance(node, ast.ImportFrom)]
                                                    for name, text in files.items() if name.endswith(".py")},
                                   "tenant_rows": counts, "runtime_context": self.memory_context(),
                                   "control_plane": json.loads((self.root / "control-plane.json").read_text()),
                                   "recent_logs": logs}, "inspection_version": self.version()}
        if tool == "deploy" and task["group"] == "ops":
            self.write_bundle("deployed", self.bundle())
            return {"deployed_version": self.version("deployed")}
        if tool == "patch" and task["group"] == "dev":
            patches = action.get("patches", [])
            if not isinstance(patches, list) or not 1 <= len(patches) <= 12:
                raise ValueError("patch requires 1..12 file edits")
            files = self.bundle()
            original = dict(files)
            for patch in patches:
                name, old, new = patch["path"], patch["old"], patch["new"]
                if name not in {module + ".py" for module in FUNCTIONS} or not old or old not in files[name]:
                    raise ValueError("patch path or exact source does not match")
                files[name] = files[name].replace(old, new)
            self.write_bundle("source", files)
            changed = sorted(name for name in files if files[name] != original[name])
            self.patch_files += len(changed)
            return {"source_version": self.version(), "changed_files": changed}
        if tool == "probe":
            request = action.get("request", task["public_case"]["request"])
            if request != task["public_case"]["request"]:
                raise ValueError("probe must use the current incident reproduction suite")
            target = "source" if task["group"] == "dev" else "deployed"
            suite = self._suite(target, task["incident"])
            self.probe_operations += suite["operation_count"]
            with (self.root / "observations.jsonl").open("a") as log:
                log.write(json.dumps({"task_id": task["task_id"], "version": self.version(target),
                                      "context": self.memory_context(), "suite": suite}) + "\n")
            return {"request": request, "actual": suite["passed"], "suite": suite,
                    "target": target, "version": self.version(target)}
        raise ValueError("tool not permitted for this project task")

    def check(self, task, observations):
        target = "source" if task["group"] == "dev" else "deployed"
        checks = []
        for index, (incident, _) in enumerate(INCIDENTS[:task["index"] + 1]):
            suite = self._suite(target, incident)
            reproduce = task["group"] == "ops" and index == task["index"]
            checks.append({"incident": incident, "criterion": "reproduce_incident" if reproduce else "regression_correct",
                           "behavior_correct": suite["passed"],
                           "passed": (not suite["passed"] and not suite["runtime_errors"]) if reproduce else suite["passed"],
                           "suite": suite})
        passed = all(row["passed"] for row in checks)
        if task["group"] == "ops":
            passed = passed and self.version() == self.version("deployed") and any(
                o.get("version") == self.version(target) and o.get("target") == target
                and o.get("request") == task["public_case"]["request"] and o.get("actual") is False
                for o in observations)
        return {"passed": passed, "checks": checks, "version": self.version(target), "runtime_context": self.memory_context(),
                "objective": "deployment_and_incident_reproduction" if task["group"] == "ops" else "cross_module_repair_and_regression"}

    def evaluate(self, task, observations):
        result = self.check(task, observations)
        target = "source" if task["group"] == "dev" else "deployed"
        hidden_checks = []
        for index, (incident, _) in enumerate(INCIDENTS[:task["index"] + 1]):
            suite = self._suite(target, incident, hidden=True)
            reproduce = task["group"] == "ops" and index == task["index"]
            hidden_checks.append({"incident": incident,
                                  "passed": (not suite["passed"] and not suite["runtime_errors"]) if reproduce else suite["passed"]})
        result["holdout_checks"] = hidden_checks
        result["passed"] = result["passed"] and all(check["passed"] for check in hidden_checks)
        return result


class ScriptedProject(ScriptedDevOps):
    def decide(self, role, context):
        if role == "planner":
            tools = context["allowed_tools"]
            if not tools:
                return {"route": "verify"}
            return {"route": "execute", "tool": tools[0]}
        if role == "executor":
            tool = context["action"]["tool"]
            if tool == "patch":
                return {"tool": "patch", "patches": deepcopy(PATCHES[context["task"]["incident"]]),
                        "diagnosis": "Scripted reference recipe; not model reasoning evidence."}
            return {"tool": tool}
        return super().decide(role, context)
