"""Bounded plan refinement driven only by public contracts and model reviews."""

import hashlib
import json

from .calculator import expression_names
from .plan_validation import validate_plan_expression, validate_quantity_contract


class PlanRepairError(ValueError):
    """Generation/checking failed within the configured correction budget."""


def plan_issues(value, *, catalog, inputs, question):
    issues = []
    if not isinstance(value, dict) or any(not isinstance(value.get(key), str) or not value[key].strip()
                                           for key in ("search_query", "expression")):
        return ["return nonempty string fields search_query and expression"]
    try:
        _, names = expression_names(value["expression"])
        unknown = names - set(catalog) - set(inputs)
        if unknown:
            issues.append(f"variables outside the entity schema: {sorted(unknown)}")
    except ValueError as exc:
        return [str(exc)]
    for check, args in ((validate_plan_expression, (inputs,)),
                        (validate_quantity_contract, (question,))):
        try:
            check(value["expression"], *args)
        except ValueError as exc:
            issues.append(str(exc))
    return issues


def repair_fingerprint(*, prompt, catalog, inputs, question, plan_prompt, review_prompt,
                       max_new_tokens, max_repairs, policy_context=None):
    return hashlib.sha256(json.dumps({
        "prompt": prompt, "catalog": catalog, "inputs": inputs, "question": question,
        "plan_prompt": plan_prompt, "review_prompt": review_prompt,
        "max_new_tokens": max_new_tokens, "max_repairs": max_repairs,
        "policy_context": policy_context, "checker_version": "quantity-contract-v1",
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _visible_record(record):
    """Do not expose persistence bookkeeping to LLM correction prompts."""
    keys = ("attempt", "kind", "candidate", "issues", "generation_error", "accepted")
    value = {key: record[key] for key in keys if key in record}
    if record.get("status") == "started":
        value["issues"] = [*value.get("issues", []), "Previous attempt was reserved but its result was not saved; that budget is already spent."]
    return value


def repair_plan(llm, *, prompt, catalog, inputs, question, plan_prompt, review_prompt,
                max_new_tokens, max_repairs, attempts, ledger=None, task_id=None, policy_context=None):
    """At most 1 + max_repairs reservations; uncertain calls are never retried.

    Completed accepted results replay without a model call. One review of a valid
    draft still checks scope; later reviews run only while checks report issues.
    """
    if isinstance(max_repairs, bool) or not isinstance(max_repairs, int) or not 1 <= max_repairs <= 5:
        raise ValueError("max_repairs must be an integer between 1 and 5")
    if ledger is not None and not task_id:
        raise ValueError("persistent repair needs task_id")
    fingerprint = repair_fingerprint(prompt=prompt, catalog=catalog, inputs=inputs, question=question,
        plan_prompt=plan_prompt, review_prompt=review_prompt, max_new_tokens=max_new_tokens,
        max_repairs=max_repairs, policy_context=policy_context)
    existing = ledger.begin(task_id, max_repairs=max_repairs, prompt_fingerprint=fingerprint)[0] if ledger is not None else []
    if existing:
        if attempts:
            raise ValueError("attempts must start empty when restoring a ledger")
        attempts.extend(_visible_record(record) for record in existing)
        accepted = next((record for record in existing if record["status"] == "accepted"), None)
        if accepted is not None:
            candidate = accepted.get("candidate")
            if plan_issues(candidate, catalog=catalog, inputs=inputs, question=question):
                raise ValueError("persisted accepted plan no longer passes checks")
            return candidate
    latest = next((record for record in reversed(attempts) if "candidate" in record), {})
    candidate = latest.get("candidate")
    issues = attempts[-1].get("issues", []) if attempts else []
    start = len(existing)
    for number in range(start, max_repairs + 1):
        if number == 0:
            messages = [{"role": "system", "content": plan_prompt}, {"role": "user", "content": prompt}]
        else:
            context = (f"\nCurrent draft: {json.dumps(candidate)}\n"
                       f"Checks to address: {json.dumps(issues) if issues else 'Check task scope, exclusions and quantities.'}\n"
                       f"Earlier attempts: {json.dumps(attempts)}\n"
                       "Fix the stated issues; do not repeat an invalid expression. "
                       "replicas is the total copy count and already includes the original.")
            messages = [{"role": "system", "content": review_prompt}, {"role": "user", "content": prompt + context}]
        record = {"attempt": number, "kind": "draft" if number == 0 else "repair"}
        reservation = None
        if ledger is not None:
            reservation = ledger.reserve(task_id, max_repairs=max_repairs, prompt_fingerprint=fingerprint,
                                          payload={**record, "issues": list(issues)})
            if reservation["attempt"] != number:
                raise ValueError("concurrent repair changed the expected attempt; dispatch refused")
        attempts.append(record)
        try:
            candidate = llm.generate_json(messages, max_new_tokens=max_new_tokens)["value"]
            record["candidate"] = candidate
            issues = plan_issues(candidate, catalog=catalog, inputs=inputs, question=question)
        except ValueError as exc:
            candidate = None
            issues = [f"generation/JSON error: {str(exc)[:1000]}"]
            record["generation_error"] = issues[0]
        record["issues"] = list(issues)
        accepted = number >= 1 and not issues
        if accepted:
            record["accepted"] = True
        if ledger is not None:
            ledger.finish(task_id, reservation, payload=record, accepted=accepted)
        if accepted:
            return candidate
    raise PlanRepairError(f"plan correction budget exhausted after {max_repairs} repairs: {'; '.join(issues)}")
