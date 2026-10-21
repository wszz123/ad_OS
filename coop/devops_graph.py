"""Bounded four-role development/operations graph with versioned shared memory."""

from datetime import datetime, timezone
from dataclasses import asdict
import json
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from .devops_world import RECIPES, digest
from .memory import MemoryItem
from .protocol import Capability, Message
from .state_exchange import StateExchange


class TaskState(TypedDict, total=False):
    task: dict
    steps: int
    route: str
    source: str
    accepted: list
    observations: list
    action: dict
    verification: dict
    status: str
    error: str
    summary: str
    state_id: str
    guide: str
    memory_candidate: dict


class ScriptedDevOps:
    """Transparent reference recipes, NOT LLM evidence or a benchmark solver."""
    def __init__(self):
        self.call_stats = []

    def decide(self, role, context):
        if role == "planner":
            task, observations = context["task"], context["observations"]
            if task["group"] == "ops" and not any("deployed_version" in o for o in observations):
                value = {"route": "execute", "tool": "deploy"}
            elif task["group"] == "dev" and not any("source_version" in o for o in observations):
                value = {"route": "execute", "tool": "patch"}
            elif not any("actual" in o for o in observations):
                value = {"route": "execute", "tool": "probe"}
            else:
                value = {"route": "verify"}
        elif role == "executor":
            tool = context["action"]["tool"]
            value = {"tool": tool}
            if tool == "patch":
                value.update(zip(("old", "new"), RECIPES[context["task"]["incident"]]))
        else:
            value = {"accepted": bool(context["verification"]["passed"]),
                     "summary": "Scripted review of current-version probes and regression results."}
        return value


class QwenDevOps:
    """Reuse a single causal model; each role gets separate bounded context."""
    def __init__(self, llm):
        self.llm = llm
        self.prompt_records = []

    @property
    def call_stats(self):
        return self.llm.call_stats

    def decide(self, role, context):
        project = context["task"].get("source_format") == "file_bundle"
        if project and role == "executor" and context["action"]["tool"] in ("inspect", "deploy", "probe"):
            # Planner already selected an argument-free capability. Mapping it to
            # the executor's typed tool call needs no second model generation.
            action = {"tool": context["action"]["tool"]}
            self.prompt_records.append({"role": role, "deterministic_binding": action})
            return action
        if project:
            from .devops_context import project_context
            context = project_context(context)
        instructions = {
            "planner": ('Return {"route":"execute","tool":"deploy|patch|probe"}, '
                        '{"route":"retrieve"}, or {"route":"verify"}. '
                        'Ops must deploy then reproduce the supplied incident; dev must repair then probe. '
                        'Choose a tool only from allowed_tools. When allowed_tools is empty, choose verify. '
                        'verify is legal ONLY when ready_for_verification=true. '
                        'prior_experience describes OTHER tasks, not actions completed in this task. '
                        'deployment_completed=true means deploy already succeeded; NEVER repeat it. '
                        'Use observations to avoid repeating successful actions. Do not repair in ops.'),
            "executor": ('Return a tool object: {"tool":"deploy"}, {"tool":"probe","request":{...}}, '
                         'or {"tool":"patch","diagnosis":"brief cause and affected types",'
                         '"old":"exact existing text","new":"replacement"}. '
                         'Use EXACTLY action.tool, never substitute a different tool. '
                         'For probe, copy task.public_case.request verbatim, including case and spaces. '
                         'Patch only the current incident. '
                         'Preserve existing defaults and unrelated predicates. If an input accepts multiple '
                         'types, normalize it before using type-specific methods. Do not repeat an edit '
                         'whose resulting version already failed a probe. '
                         'Source must remain one search(records, request) function using simple '
                         'assignments, expressions, if statements and list comprehensions; no imports.'),
            "verifier": ('Return {"accepted":true|false,"summary":"evidence-backed conclusion"}. '
                         'Ops success means a reproduced unresolved incident and verified previous repairs; '
                         'dev success means passing current and previous public checks. '
                         'verification.passed is the executable public gate: for ops an expected mismatch '
                         'is a successful reproduction, NOT a repair failure. '
                         'Never claim repair merely because a patch was written.'),
        }
        if project:
            instructions["planner"] = instructions["planner"].replace("deploy|patch|probe", "inspect|deploy|patch|probe")
            instructions["planner"] += (
                ' For patch actions include files (candidate module paths) and goal. '
                'Cover every independent public failure, not just the most recent cache symptom. '
                'Use the import graph and function contracts; prior task success is not current evidence.')
            instructions["executor"] = (
                'Return exactly action.tool. inspect/deploy/probe use {"tool":"..."}; probe runs the task public scenario. '
                'patch uses {"tool":"patch","diagnosis":"brief cross-module cause",'
                '"patches":[{"path":"module.py","old":"exact existing text","new":"replacement"}, ...]}. '
                'All edits apply atomically. Inspect docs/architecture.md and docs/policy.md and preserve prior requirements. '
                'Preserve each module function signature. Only local from-imports, assignments, if/for statements, '
                'comprehensions and basic dict/list/string operations are allowed; no filesystem, network, or system imports. '
                'Use public probe/check feedback and action.files/goal to repair all affected modules. '
                'Only use parameters, assigned locals, declared imports and permitted builtins; do not invent globals. '
                'Do not repeat known unsuccessful edits. Different independent failures can require edits in different modules.')
        constraint = ""
        if role == "executor":
            constraint = "\nRequired tool for THIS call: " + context["action"]["tool"]
        elif role == "planner":
            constraint = ("\nAllowed tools NOW: " + json.dumps(context["allowed_tools"])
                          + "\nAllowed routes NOW: " + json.dumps(context["allowed_routes"]))
        # Code is rendered as code, rather than a JSON string full of escaped
        # newlines. Both transports still supply the same underlying evidence.
        prompt_context = {key: value for key, value in context.items() if key != "source"}
        content = json.dumps(prompt_context, ensure_ascii=False)
        if "source" in context:
            if context["task"].get("source_format") == "file_bundle":
                for name, text in json.loads(context["source"]).items():
                    content += "\nFile: " + name + "\n```\n" + text + "\n```\n"
            else:
                content += "\nCurrent source:\n```python\n" + context["source"] + "```"
        token_budget = 2048 if role == "executor" and context["task"].get("source_format") == "file_bundle" else 768
        messages = [
            {"role": "system", "content": instructions[role]},
            {"role": "user", "content": content + constraint},
        ]
        self.prompt_records.append({"role": role, "max_new_tokens": token_budget, "messages": messages})
        return self.llm.generate_json(messages, max_new_tokens=token_budget)["value"]


def build_devops_graph(world, memory, transport, decision, encoder, *, run_id,
                       memory_scope="shared", use_vector=True, max_steps=12,
                       checkpointer=None, corpus=None):
    if memory_scope not in ("none", "group", "shared"):
        raise ValueError("memory scope must be none, group or shared")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    exchange, packets, metrics = StateExchange(), {}, {}
    project_id = getattr(world, "project_id", "catalog-service")

    def runtime_context():
        return world.memory_context() if hasattr(world, "memory_context") else {}

    def call(role, payload):
        # Both transports receive exactly the same semantic fields.
        result = transport.dispatch(Message("runtime", role, role,
                                             {"context": json.dumps(payload, ensure_ascii=False)}))
        return json.loads(result["result"])

    def retrieve(context):
        task = context["task"]
        event = metrics[task["task_id"]]
        query = "Service search code incident diagnosis repair " + task["instruction"]
        state_id = context["state_id"]
        vector = exchange.decode(packets[state_id]) if state_id else None
        guide = ""
        if corpus is not None:
            matches = (corpus.search_vector(vector, limit=1) if vector is not None
                       else corpus.search("catalog" if context["task"].get("source_format") == "file_bundle" else "search", limit=1))
            guide = matches[0].item.summary if matches else ""
        accepted, rejected = [], []
        version = world.version()
        if memory_scope != "none":
            event["memory_queries"] += 1
            tags = ["run:" + run_id]
            if memory_scope == "group":
                tags.append("group:" + task["group"])
            candidates = (memory.search_vector(vector, tags=tags, limit=100) if vector is not None
                          else memory.search("search", tags=tags, limit=100))
            event["memory_candidates"] += len(candidates)
            for candidate in sorted(candidates, key=lambda c: c.item.created_at, reverse=True):
                item = candidate.item
                record = json.loads(item.evidence)
                if (record["source_version"] != version or digest(record["source"]) != version
                        or not record["verified"]
                        or record["project_id"] != project_id
                        or record["task_id"] == task["task_id"]):
                    rejected.append(item.memory_id)
                    continue
                accepted.append({"memory_id": item.memory_id, **record})
            event["rejected_memory_ids"].extend(rejected)
        if accepted:
            selected = dict(accepted[0])
            if selected.get("runtime_context", {}) != runtime_context():
                selected["observations"] = []
                selected["summary"] = "Verified code snapshot only; runtime/data/policy context changed. Reinspect current evidence."
                event["context_filtered_reuse"] += 1
            source = selected["source"]
            event["accepted_reuse"] += 1
            event["avoided_source_reads"] += 1
            event["used_memory_ids"].append(selected["memory_id"])
            event["cross_group_reuse"] += int(selected["group"] != task["group"])
            # Passing source evidence downstream constitutes use; correctness is graded separately.
            accepted = [selected]
        else:
            source = world.read_source()
        return {"source": source, "accepted": accepted, "guide": guide}

    transport.register(Capability("runtime", "runtime", ()), {})
    for role in ("planner", "executor", "verifier"):
        def handler(params, role=role):
            return {"result": json.dumps(decision.decide(role, json.loads(params["context"])))}
        transport.register(Capability(role, role, (role,)), {role: handler})
    transport.register(Capability("retriever", "retrieval", ("retriever",)), {
        "retriever": lambda p: {"result": json.dumps(retrieve(json.loads(p["context"])))}})

    def initialize(state):
        task = state["task"]
        metrics[task["task_id"]] = {
            "memory_queries": 0, "memory_candidates": 0, "accepted_reuse": 0,
            "cross_group_reuse": 0, "avoided_source_reads": 0, "state_transfers": 0,
            "state_payload_bytes": 0, "state_metadata_bytes": 0,
            "context_filtered_reuse": 0,
            "used_memory_ids": [], "rejected_memory_ids": [],
        }
        state_id = ""
        if use_vector:
            query = "Service search code incident diagnosis repair " + task["instruction"]
            vector = encoder([query], query=True)[0]
            state_id = task["task_id"] + ":query:" + uuid4().hex[:8]
            packet = exchange.encode(vector, state_id=state_id,
                                     source_agent="planner", target_agent="retriever")
            packets[state_id] = packet
            metrics[task["task_id"]]["state_transfers"] += 1
            metrics[task["task_id"]]["state_payload_bytes"] += packet.payload_bytes
            metadata = asdict(packet)
            metadata.pop("payload")
            metrics[task["task_id"]]["state_metadata_bytes"] += len(
                json.dumps(metadata, separators=(",", ":")).encode())
        return {"steps": 0, "observations": [], "accepted": [], "status": "running",
                "source": "", "error": "", "verification": {}, "summary": "", "state_id": state_id,
                "memory_candidate": {}}

    def retrieval(state):
        return {**call("retriever", {"task": state["task"], "state_id": state["state_id"]}),
                "steps": state["steps"] + 1}

    def planner(state):
        if state["steps"] >= max_steps:
            return {"status": "budget_exhausted", "route": "finish"}
        context = {key: state[key] for key in ("task", "source", "guide", "observations", "verification")}
        context["memory_refs"] = [m["memory_id"] for m in state["accepted"]]
        context["prior_experience"] = [{key: m[key] for key in ("task_id", "group", "summary", "observations")}
                                       for m in state["accepted"]]
        context["deployment_completed"] = any("deployed_version" in o for o in state["observations"])
        target = "source" if state["task"]["group"] == "dev" else "deployed"
        mutation_key = "source_version" if target == "source" else "deployed_version"
        last_mutation = max((i for i, o in enumerate(state["observations"]) if mutation_key in o), default=-1)
        probes = [o for i, o in enumerate(state["observations"]) if i > last_mutation
                  and ("actual" in o or o.get("tool") == "probe")
                  and o.get("request") == state["task"]["public_case"]["request"]
                  and o.get("version") == world.version(target)]
        context["current_version_probed"] = bool(probes)
        patched = any("source_version" in o for o in state["observations"])
        failed_verification = (state["verification"].get("passed") is False
                               and state["verification"].get("version") == world.version(target))
        if state["task"]["group"] == "dev":
            # Every mutation must be tested before another mutation. A passing probe
            # leads to verification, not repeated speculative edits or deployment.
            context["allowed_tools"] = ([] if probes and not failed_verification
                                        and probes[-1].get("actual") == state["task"]["public_case"]["expected"]
                                        else ["patch"] if probes else ["probe"] if patched else ["patch", "probe"])
        else:
            context["allowed_tools"] = (["deploy"] if not context["deployment_completed"]
                                        else [] if probes else ["probe"])
        if getattr(world, "requires_inspection", False) and not any("inspection" in o for o in state["observations"]):
            context["allowed_tools"] = ["inspect"]
        context["ready_for_verification"] = not context["allowed_tools"]
        context["allowed_routes"] = ["retrieve", "verify" if context["ready_for_verification"] else "execute"]
        try:
            action = call("planner", context)
            # Explicit schema mapping: some tool-capable models return a tool name
            # as the route. Preserve the original response in the protocol trace.
            if action.get("route") in ("inspect", "deploy", "patch", "probe") and "tool" not in action:
                action = {**action, "route": "execute", "tool": action["route"], "mapped_from": action}
            route = action.get("route")
            if route not in ("execute", "retrieve", "verify"):
                raise ValueError("planner returned unsupported route")
            if route not in context["allowed_routes"]:
                raise ValueError("current task needs tool evidence before verification; history is not current evidence")
            if route == "execute" and action.get("tool") not in context["allowed_tools"]:
                raise ValueError("planner selected a tool outside the current capability constraints")
            if "files" in action and state["task"].get("source_format") == "file_bundle":
                editable = {name for name in json.loads(state["source"]) if name.endswith(".py")}
                if not isinstance(action["files"], list) or any(not isinstance(name, str) or name not in editable for name in action["files"]):
                    raise ValueError("repair plan files must be current project module paths")
            if state["steps"] + 2 > max_steps:
                return {"status": "budget_exhausted", "route": "finish", "steps": state["steps"] + 1}
            return {"route": route, "action": action, "steps": state["steps"] + 1}
        except Exception as exc:
            return {"route": "replan", "steps": state["steps"] + 1,
                    "observations": [*state["observations"], {"planner_error": str(exc)}]}

    def executor(state):
        action = state["action"]
        try:
            action = call("executor", {key: state[key] for key in ("task", "source", "action", "observations", "verification")})
            if action.get("tool") != state["action"].get("tool"):
                raise ValueError("executor changed the selected tool")
            result = world.execute(state["task"], action)
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
            if action.get("tool") == "probe":
                target = "source" if state["task"]["group"] == "dev" else "deployed"
                result.update(tool="probe", target=target, version=world.version(target),
                              request=action.get("request", state["task"]["public_case"]["request"]))
        # Source updates are returned to the executor/planner through tool observations.
        source = world.source.read_text(encoding="utf-8") if "source_version" in result else state["source"]
        return {"source": source, "observations": [*state["observations"], result],
                "steps": state["steps"] + 1}

    def verifier(state):
        try:
            verification = world.check(state["task"], state["observations"])
            review = call("verifier", {"task": state["task"], "verification": verification,
                                       "observations": state["observations"]})
            passed = verification["passed"] and review.get("accepted") is True
            return {"verification": verification, "summary": str(review.get("summary", "")),
                    "status": "verified" if passed else "running", "steps": state["steps"] + 1}
        except Exception as exc:
            return {"verification": {"passed": False, "error": str(exc)},
                    "steps": state["steps"] + 1}

    def finish(state):
        if state["status"] == "verified" and memory_scope != "none":
            task = state["task"]
            summary = (f"{task['task_id']} ops: deployed and reproduced {task['incident']}; "
                       "the current incident remains unresolved, previous regressions passed."
                       if task["group"] == "ops" else
                       f"{task['task_id']} dev: candidate repair for {task['incident']} passed public regression checks; "
                       "candidate is not yet deployed.")
            record = {"project_id": project_id, "task_id": task["task_id"], "group": task["group"],
                      "source_version": world.version(), "source": state["source"], "verified": True,
                      "observations": state["observations"], "summary": summary,
                      "runtime_context": runtime_context(),
                      "summary_origin": "executable_task_objective"}
            query = "Service search code incident diagnosis repair " + task["instruction"]
            vector = encoder([query])[0]
            item = MemoryItem(
                run_id + ":" + task["task_id"], "verifier", datetime.now(timezone.utc).isoformat(),
                project_id, "search source and verified " + task["instruction"],
                memory_type="verified_source_and_experience", tags=["run:" + run_id, "group:" + task["group"]],
                evidence=json.dumps(record), embedding=list(map(float, vector)))
            # Scheduler commits only after independent grading; hidden failures never
            # seed subsequent tasks, and hidden details are never fed to this graph.
            return {"memory_candidate": asdict(item)}
        return {}

    builder = StateGraph(TaskState)
    for name, fn in (("initialize", initialize), ("retrieve", retrieval), ("plan", planner),
                     ("execute", executor), ("verify", verifier), ("finish", finish)):
        builder.add_node(name, fn)
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "retrieve")
    builder.add_edge("retrieve", "plan")
    builder.add_conditional_edges("plan", lambda s: s["route"],
                                  {"execute": "execute", "retrieve": "retrieve",
                                   "verify": "verify", "finish": "finish", "replan": "plan"})
    builder.add_edge("execute", "plan")
    builder.add_conditional_edges("verify", lambda s: "finish" if s["status"] == "verified" else "plan",
                                  {"finish": "finish", "plan": "plan"})
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer), packets, metrics
