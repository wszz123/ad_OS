"""Versioned local service exercise. Constructed workload, not a public benchmark."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


SOURCE = '''def search(records, request):
    query = request.get("query", "").lower()
    offset = request.get("offset") or 1
    limit = int(request.get("limit", 20))
    active = bool(request.get("active", True))
    matches = [r for r in records if query in r["name"].lower() and r["active"] == active]
    return [r["id"] for r in matches[offset:offset + limit]]
'''

RECORDS = [
    {"id": 1, "name": "Alpha", "active": True},
    {"id": 2, "name": "Alpha archived", "active": False},
    {"id": 3, "name": "Straße", "active": True},
    {"id": 4, "name": "Alphabet", "active": True},
]

# Reference edits belong exclusively to the explicitly scripted decision fixture.
RECIPES = {
    "query": ('request.get("query", "").lower()', 'request.get("query", "").strip().lower()'),
    "offset": ('request.get("offset") or 1', 'int(request.get("offset", 0))'),
    "limit": ('int(request.get("limit", 20))', 'max(0, min(100, int(request.get("limit", 20))))'),
    "active": ('bool(request.get("active", True))', 'str(request.get("active", True)).lower() == "true"'),
    "unicode": ('.lower()', '.casefold()'),
}

INCIDENTS = [
    ("query", "Search must ignore leading and trailing query whitespace.",
     {"query": " Alpha ", "offset": 1}, [4]),
    ("offset", "Offset zero must include the first match; offset defaults to zero.",
     {"query": "Alpha", "offset": 0}, [1, 4]),
    ("limit", "Limit is clamped to the inclusive range 0..100; negative means no results.",
     {"query": "", "offset": 0, "limit": -1}, []),
    ("active", "The active filter accepts booleans and the strings true/false; it defaults to true when omitted.",
     {"query": "Alpha", "offset": 0, "active": "false"}, [2]),
    ("unicode", "Search must use Unicode caseless matching, including STRASSE versus Straße.",
     {"query": "STRASSE", "offset": 0}, [3]),
]


def tasks():
    sequence = []
    for index, (key, requirement, request, expected) in enumerate(INCIDENTS):
        for group in ("ops", "dev"):
            task_id = ("B" if group == "ops" else "A") + str(index + 1)
            sequence.append({
                "task_id": task_id, "group": group, "incident": key, "index": index,
                "depends_on": [sequence[-1]["task_id"]] if sequence else [],
                "instruction": (("Deploy the candidate, reproduce and document this incident; "
                                 "confirm previously repaired behavior still works. " if group == "ops" else
                                 "Repair this incident in source, preserve previously repaired behavior, "
                                 "and produce a verified handoff for operations. ") + requirement),
                "requirements": [item[1] for item in INCIDENTS[:index + 1]],
                "public_case": {"request": request, "expected": expected},
            })
    return sequence


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class DevOpsWorld:
    """Allowlisted file edits and isolated, timed expression evaluation.

    Generated source is validated against a deliberately small Python subset.
    This is not a general CodeAct/container sandbox or a production service.
    """

    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.source = self.root / "candidate.py"
        self.deployed = self.root / "deployed.py"
        for path in (self.source, self.deployed):
            if not path.exists():
                path.write_text(SOURCE, encoding="utf-8")
        self.source_reads = 0
        self.tool_calls = 0

    def version(self, target="source"):
        return digest((self.source if target == "source" else self.deployed).read_text(encoding="utf-8"))

    def read_source(self):
        self.source_reads += 1
        return self.source.read_text(encoding="utf-8")

    @staticmethod
    def validate(source):
        tree = ast.parse(source)
        allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.Return,
                   ast.Name, ast.Load, ast.Store, ast.Constant, ast.Call, ast.Attribute,
                   ast.ListComp, ast.comprehension, ast.Subscript, ast.Slice, ast.BinOp,
                   ast.Add, ast.UnaryOp, ast.USub, ast.Compare, ast.In, ast.Eq, ast.BoolOp,
                   ast.And, ast.Or, ast.List, ast.Tuple, ast.IfExp, ast.NotEq, ast.If,
                   ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Not, ast.Is, ast.IsNot)
        if len(source) > 6000 or len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError("only one bounded search function is permitted")
        fn = tree.body[0]
        if (fn.name != "search" or [a.arg for a in fn.args.args] != ["records", "request"]
                or fn.decorator_list or fn.args.defaults or fn.returns):
            raise ValueError("search(records, request) must retain its signature")
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                raise ValueError(f"unsupported syntax: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id.startswith("_"):
                raise ValueError("private names are not permitted")
            if isinstance(node, ast.Attribute) and node.attr not in {"get", "lower", "casefold", "strip"}:
                raise ValueError("unsupported attribute")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id not in {"int", "str", "bool", "min", "max", "isinstance"}:
                    raise ValueError("unsupported function")
                if not isinstance(node.func, (ast.Name, ast.Attribute)):
                    raise ValueError("unsupported call")

    def probe(self, target, request, records=None):
        if target not in ("source", "deployed"):
            raise ValueError("unknown target")
        source = (self.source if target == "source" else self.deployed).read_text(encoding="utf-8")
        self.validate(source)
        # No generated imports or unrestricted builtins. Child owns no persistent worker state.
        runner = ("import json,sys\n"
                  "data=json.load(sys.stdin)\n"
                  "scope={'__builtins__':{k:__builtins__.__dict__[k] for k in ['int','str','bool','min','max','isinstance']}}\n"
                  "exec(data['source'],scope)\n"
                  "print(json.dumps(scope['search'](data['records'],data['request'])))\n")
        result = subprocess.run([sys.executable, "-I", "-c", runner],
                                input=json.dumps({"source": source, "request": request,
                                                  "records": RECORDS if records is None else records}),
                                text=True, capture_output=True, timeout=3, cwd=self.root)
        if result.returncode:
            raise ValueError(result.stderr[-1500:])
        return json.loads(result.stdout)

    def execute(self, task, action):
        self.tool_calls += 1
        name = action.get("tool")
        if name == "deploy" and task["group"] == "ops":
            self.deployed.write_text(self.source.read_text(encoding="utf-8"), encoding="utf-8")
            return {"deployed_version": self.version("deployed")}
        if name == "patch" and task["group"] == "dev":
            old, new = action["old"], action["new"]
            source = self.source.read_text(encoding="utf-8")
            if not old or old not in source:
                raise ValueError("patch does not match current source")
            revised = source.replace(old, new)
            self.validate(revised)
            self.source.write_text(revised, encoding="utf-8")
            return {"source_version": self.version()}
        if name == "probe":
            target = "deployed" if task["group"] == "ops" else "source"
            request = action.get("request", task["public_case"]["request"])
            return {"request": request, "actual": self.probe(target, request),
                    "version": self.version(target), "target": target}
        raise ValueError("tool not permitted for this task")

    def check(self, task, observations):
        target = "deployed" if task["group"] == "ops" else "source"
        rows = []
        for index in range(task["index"] + 1):
            _, _, request, expected = INCIDENTS[index]
            try:
                actual, error = self.probe(target, request), ""
            except Exception as exc:
                actual, error = None, str(exc)
            reproduce = task["group"] == "ops" and index == task["index"]
            rows.append({"incident": INCIDENTS[index][0], "actual": actual,
                         "expected_healthy": expected, "behavior_correct": actual == expected,
                         "criterion": "reproduce_incident" if reproduce else "regression_correct",
                         "error": error,
                         "passed": not error and (actual != expected if reproduce else actual == expected)})
        if task["group"] == "dev":
            passed = all(row["passed"] for row in rows)
        else:
            # A diagnosis task must expose the incident on the current deployment,
            # not pretend to have repaired it. Earlier repairs must remain working.
            passed = (self.version() == self.version("deployed")
                      and all(row["passed"] for row in rows)
                      and any(item.get("target") == "deployed"
                              and item.get("version") == self.version("deployed")
                              and item.get("request") == task["public_case"]["request"]
                              and item.get("actual") == rows[-1]["actual"] for item in observations))
        boundary_checks = []
        def boundary(request, expected, criterion, records=None):
            try:
                actual = self.probe(target, request, records)
                return {"criterion": criterion, "actual": actual, "expected": expected, "passed": actual == expected}
            except Exception as exc:
                return {"criterion": criterion, "passed": False, "error": str(exc)}
        repaired_through = task["index"] if task["group"] == "dev" else task["index"] - 1
        if repaired_through >= 1:
            boundary_checks.append(boundary({"query": "Alpha"}, [1, 4], "offset_default_zero"))
        if repaired_through >= 2:
            many = [{"id": i, "name": "Alpha", "active": True} for i in range(1, 111)]
            boundary_checks.append(boundary({"offset": 0, "limit": 1000}, list(range(1, 101)), "limit_upper_bound", many))
        if repaired_through >= 3:
            boundary_checks.append(boundary({"query": "Alpha", "offset": 0, "active": False}, [2], "boolean_false_filter"))
        passed = passed and all(row["passed"] for row in boundary_checks)
        return {"passed": passed, "checks": rows, "boundary_checks": boundary_checks, "version": self.version(target),
                "objective": "deployment_and_incident_reproduction" if task["group"] == "ops" else "repair_and_regression"}

    def evaluate(self, task, observations):
        """Independent, post-task holdout checks; never fed back to agents."""
        result = self.check(task, observations)
        def matches(request, expected, records):
            try:
                return self.probe("source", request, records) == expected
            except Exception:
                return False
        if task["group"] == "dev":
            records = [{"id": 11, "name": "Straße", "active": True},
                       {"id": 12, "name": "Beta", "active": False},
                       {"id": 13, "name": "Betamax", "active": True},
                       {"id": 14, "name": "Beta second", "active": True}]
            cases = [({"query": " Beta ", "offset": 0}, [13]),
                     ({"query": "", "offset": 0}, [11, 13, 14]),
                     ({"query": "", "offset": 0, "limit": -2}, []),
                     ({"query": "Beta", "offset": 0, "active": "false"}, [12]),
                     ({"query": "STRASSE", "offset": 0}, [11])]
            # First case avoids the still-unfixed offset behavior in A1.
            cases[0] = ({"query": " Beta ", "offset": 1}, [14])
            hidden = [matches(request, expected, records)
                      for request, expected in cases[:task["index"] + 1]]
            if task["index"] >= 1:
                hidden.append(matches({"query": "Beta"}, [13, 14], records))
            if task["index"] >= 2:
                many = [{"id": i, "name": "Beta", "active": True} for i in range(1, 106)]
                hidden.append(matches({"limit": 1000, "offset": 0}, list(range(1, 101)), many))
            if task["index"] >= 3:
                hidden.append(matches({"query": "Beta", "active": False, "offset": 0}, [12], records))
            if task["index"] >= 4:
                hidden.append(matches({"query": " strasse ", "offset": 0}, [11], records))
            result["holdout_checks"] = hidden
            result["passed"] = result["passed"] and all(hidden)
        return result
