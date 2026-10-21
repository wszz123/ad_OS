"""Public multi-module service seed and explicitly scripted fixture patches."""

FILES = {
    "repository.py": '''def select(records, tenant):
    return list(records)
''',
    "policies.py": '''def resolve(defaults, overrides):
    return {"include_archived": bool(overrides.get("include_archived") or defaults.get("include_archived", False)),
            "max_limit": int(overrides.get("max_limit") or defaults.get("max_limit", 20))}
''',
    "limits.py": '''def bound_limit(value, policy):
    return max(0, min(100, int(value or 20)))
''',
    "cache.py": '''def cache_key(request, revision, policy):
    return str((request.get("query", ""),))
''',
    "pagination.py": '''def page(rows, request, revision):
    rows = sorted(rows, key=lambda r: r["name"].casefold())
    cursor = request.get("cursor")
    if cursor:
        rows = [r for r in rows if r["name"].casefold() > cursor["name"]]
    limit = request["limit"]
    selected = rows[:limit]
    following = None
    if selected and len(rows) > limit:
        following = {"name": selected[-1]["name"].casefold()}
    return {"rows": selected, "next_cursor": following}
''',
    "mutations.py": '''def commit(records, command, revision):
    for change in command["changes"]:
        matches = [r for r in records if r["tenant"] == command["tenant"] and r["id"] == change["id"]]
        if not matches:
            return records, revision, {"error": "not_found"}
        row = matches[0]
        stock = row["stock"] + change.get("delta", 0)
        if stock < 0:
            return records, revision, {"error": "out_of_stock"}
        row["stock"] = stock
    return records, revision, {"ok": True}
''',
    "gateway.py": '''from mutations import commit

def apply(records, command, state):
    if not command.get("changes") or not command.get("idempotency_key"):
        return records, {"error": "bad_request"}
    records, revision, result = commit(records, command, state["revision"])
    state["revision"] = revision
    return records, result
''',
    "service.py": '''from repository import select
from policies import resolve
from limits import bound_limit
from cache import cache_key
from pagination import page

def search(records, request, state, defaults, overrides):
    request = dict(request)
    request["query"] = request.get("query", "").strip().casefold()
    policy = resolve(defaults, overrides)
    request["limit"] = bound_limit(request.get("limit"), policy)
    key = cache_key(request, state["revision"], policy)
    if key in state["cache"]:
        state["cache_hits"] = state["cache_hits"] + 1
        return state["cache"][key]
    state["cache_misses"] = state["cache_misses"] + 1
    rows = select(records, request["tenant"])
    rows = [r for r in rows if request["query"] in r["name"].casefold() and (policy["include_archived"] or r["active"])]
    result = page(rows, request, state["revision"])
    if "error" in result:
        return result
    response = {"items": [{"tenant": r["tenant"], "id": r["id"], "name": r["name"], "stock": r["stock"]} for r in result["rows"]],
                "next_cursor": result["next_cursor"], "revision": state["revision"]}
    state["cache"][key] = response
    return response
''',
    "docs/architecture.md": """# Catalog service v2
service.py coordinates repository, policies, limits, pagination and cache.
gateway.py coordinates mutations and persistent receipts. Repository storage is
SQLite; scenario workers load a snapshot, run the actual modules, then checkpoint
records, cache, revision and receipts. Requests require a tenant. IDs are only
unique within a tenant. Sort order is (Unicode-casefolded name, numeric ID).
Every successful write increments catalog revision once. Cursors are bound to
tenant, normalized query and revision. A stale or foreign cursor must be rejected.
Cache keys must reflect tenant, query, pagination, catalog revision and effective
policy. A cache hit is not authoritative when any of these have changed.
""",
    "docs/policy.md": """# Current contract v2
include_archived defaults to false; tenant overrides can be boolean or true/false
strings and an explicit false must not fall back to a true global default.
Default limit is 20, clamped to 0..min(100, tenant max_limit). Zero means empty.
Batch writes are all-or-nothing, including invalid IDs and insufficient stock.
Idempotency keys are tenant-scoped. Same key and same changes replay once even
after a process restart. Different changes with an existing key are rejected.
Optimistic expected_revision is checked for new writes, after receipt lookup.
The same idempotency key used by another tenant is an independent request.
""",
    "docs/legacy-v0.md": """# Obsolete API v0 (not the active contract)
Old clients used name-only cursors and a process-local query cache. These rules
must not override docs/policy.md or docs/architecture.md for v2 requests.
""",
}

PAGE_FIXED = '''def page(rows, request, revision):
    rows = sorted(rows, key=lambda r: (r["name"].casefold(), r["id"]))
    cursor = request.get("cursor")
    if cursor:
        if cursor.get("tenant") != request["tenant"] or cursor.get("query") != request["query"] or cursor.get("revision") != revision:
            return {"error": "invalid_cursor"}
        rows = [r for r in rows if (r["name"].casefold(), r["id"]) > (cursor["name"], cursor["id"])]
    limit = request["limit"]
    selected = rows[:limit]
    following = None
    if selected and len(rows) > limit:
        following = {"tenant": request["tenant"], "query": request["query"], "revision": revision,
                     "name": selected[-1]["name"].casefold(), "id": selected[-1]["id"]}
    return {"rows": selected, "next_cursor": following}
'''

POLICY_FIXED = '''def resolve(defaults, overrides):
    flag = overrides.get("include_archived", defaults.get("include_archived", False))
    return {"include_archived": str(flag).lower() == "true",
            "max_limit": int(overrides.get("max_limit", defaults.get("max_limit", 20)))}
'''

COMMIT_FIXED = '''def commit(records, command, revision):
    changed = [dict(r) for r in records]
    for change in command["changes"]:
        matches = [r for r in changed if r["tenant"] == command["tenant"] and r["id"] == change["id"]]
        if not matches:
            return records, revision, {"error": "not_found"}
        row = matches[0]
        stock = row["stock"] + change.get("delta", 0)
        if stock < 0:
            return records, revision, {"error": "out_of_stock"}
        row["stock"] = stock
    return changed, revision + 1, {"ok": True}
'''

GATEWAY_FIXED = '''from mutations import commit

def apply(records, command, state):
    if not command.get("changes") or not command.get("idempotency_key"):
        return records, {"error": "bad_request"}
    key = str((command["tenant"], command["idempotency_key"]))
    fingerprint = str([sorted(change.items()) for change in command["changes"]])
    receipt = state["receipts"].get(key)
    if receipt:
        if receipt["fingerprint"] != fingerprint:
            return records, {"error": "idempotency_conflict"}
        return records, receipt["result"]
    if command.get("expected_revision", state["revision"]) != state["revision"]:
        return records, {"error": "revision_conflict"}
    records, revision, result = commit(records, command, state["revision"])
    state["revision"] = revision
    if result.get("ok"):
        state["receipts"][key] = {"fingerprint": fingerprint, "result": result}
    return records, result
'''

KEYS = [
    'str((request.get("query", ""),))',
    'str((request["tenant"], request.get("query", "")))',
    'str((request["tenant"], request.get("query", ""), request.get("cursor"), request["limit"]))',
    'str((request["tenant"], request.get("query", ""), request.get("cursor"), request["limit"], revision))',
    'str((request["tenant"], request.get("query", ""), request.get("cursor"), request["limit"], revision, policy["include_archived"], policy["max_limit"]))',
]

# Only ScriptedProject consumes these; never included in model-visible evidence.
PATCHES = {
    "tenant_cache": [
        {"path": "repository.py", "old": "return list(records)", "new": 'return [r for r in records if r["tenant"] == tenant]'},
        {"path": "cache.py", "old": KEYS[0], "new": KEYS[1]},
    ],
    "cursor": [
        {"path": "pagination.py", "old": FILES["pagination.py"], "new": PAGE_FIXED},
        {"path": "cache.py", "old": KEYS[1], "new": KEYS[2]},
    ],
    "invalidation": [
        {"path": "mutations.py", "old": 'return records, revision, {"ok": True}', 'new': 'return records, revision + 1, {"ok": True}'},
        {"path": "cache.py", "old": KEYS[2], "new": KEYS[3]},
    ],
    "policy": [
        {"path": "policies.py", "old": FILES["policies.py"], "new": POLICY_FIXED},
        {"path": "limits.py", "old": "max(0, min(100, int(value or 20)))", "new": 'max(0, min(100, policy["max_limit"], int(20 if value is None else value)))'},
        {"path": "cache.py", "old": KEYS[3], "new": KEYS[4]},
    ],
    "transactions": [
        {"path": "mutations.py", "old": FILES["mutations.py"].replace('return records, revision, {"ok": True}', 'return records, revision + 1, {"ok": True}'), "new": COMMIT_FIXED},
        {"path": "gateway.py", "old": FILES["gateway.py"], "new": GATEWAY_FIXED},
    ],
}

INCIDENTS = [
    ("tenant_cache", "Isolate tenant-scoped IDs in repository selection AND cached responses; alternating tenants must never share results."),
    ("cursor", "Paginate deterministically across duplicate names using (casefolded name, ID); bind cursors to tenant, query and revision; include pagination in the cache key."),
    ("invalidation", "Successful writes advance catalog revision once; subsequent cached reads must reflect stock updates and old cursors must be rejected."),
    ("policy", "Honor dynamic global and tenant policy, explicit false and false strings; limit=0 is empty and tenant max_limit is enforced; cached responses must track effective policy."),
    ("transactions", "Make batch stock updates atomic, tenant-scoped and idempotent across worker restart; reject conflicting replays and stale expected_revision without side effects."),
]


def project_tasks():
    sequence = []
    for index, (incident, requirement) in enumerate(INCIDENTS):
        for group in ("ops", "dev"):
            task_id = ("B" if group == "ops" else "A") + str(index + 1)
            sequence.append({
                "task_id": task_id, "group": group, "index": index, "incident": incident,
                "workload": "constructed_catalog_project_v2", "source_format": "file_bundle",
                "depends_on": [sequence[-1]["task_id"]] if sequence else [],
                "instruction": (("Inspect deployment/config/log evidence, deploy the candidate, reproduce the incident and validate earlier repairs. "
                                 if group == "ops" else "Inspect cross-module dependencies and operations evidence; repair the incident atomically across relevant files, then validate all earlier contracts. ") + requirement),
                "requirements": [text for _, text in INCIDENTS[:index + 1]],
                "public_case": {"request": {"scenario": incident}, "expected": True},
            })
    return sequence
