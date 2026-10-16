"""Interrupt a Planner node before its checkpoint; resume its attempt ledger."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import threading
import time

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from typing import TypedDict

from coop.plan_repair import PlanRepairError, repair_plan
from coop.repair_ledger import RepairLedger
from coop.qwen_agent import QwenCausalAdapter
from coop.cost_graph import PLAN_PROMPT, REVIEW_PROMPT
from coop.workload import cost_tasks, variable_catalog


MODEL = "/home/Users/SharedFile/Models/Qwen/Qwen3-8B"
GOOD = "node_count * power_watts / 1000 * hours_per_day * days * energy_rate"
BAD = "power_watts / 1000 * hours_per_day * days * energy_rate"


class State(TypedDict, total=False):
    task_id: str
    plan: dict
    plan_attempts: list


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


class WorkerLLM:
    def __init__(self, *, fake, phase, case, device, max_tokens, stop, model_path=MODEL):
        self.fake, self.phase, self.case, self.stop = fake, phase, case, stop
        self.adapter = None if fake else QwenCausalAdapter(model_path, device=device, max_new_tokens=max_tokens)
        self.calls = 0

    def generate_json(self, messages, *, max_new_tokens):
        self.calls += 1
        if self.fake:
            expression = BAD if self.case == "exhausted" or (self.case == "inflight" and self.phase == "prepare") else GOOD
            result = {"value": {"search_query": "Orion electricity facts", "expression": expression}}
        else:
            result = self.adapter.generate_json(messages, max_new_tokens=max_new_tokens)
        if self.phase == "prepare" and self.case == "inflight" and self.calls == 1:
            # Real generation has returned, but repair_plan has not durably saved
            # its result. The reservation is the only ledger evidence of the call.
            self.stop("generation_returned_before_ledger_finish")
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--phase", choices=("prepare", "resume"), required=True)
    parser.add_argument("--case", choices=("inflight", "accepted", "exhausted"), default="inflight")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--model-path", default=MODEL)
    args = parser.parse_args()
    if args.case == "exhausted" and not args.fake:
        parser.error("forced exhaustion uses fake responses; real tests do not alter model outputs")
    directory = Path(args.directory)
    started = time.perf_counter()
    task = cost_tasks()[1]
    catalog = variable_catalog(task.entity)
    prompt = (f"Entity: {task.entity}\nTask: {task.question}\n"
              f"Source variable meanings and units (values must be retrieved): {json.dumps(catalog, sort_keys=True)}\n"
              f"User inputs: {json.dumps(task.inputs, sort_keys=True)}\nUnit: {task.unit}")
    with ExitStack() as stack:
        ledger = stack.enter_context(RepairLedger(str(directory / "repair_ledger.sqlite")))
        saver = stack.enter_context(SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")))
        def stop(boundary):
            marker = {"pid": os.getpid(), "case": args.case, "boundary": boundary,
                      "calls": llm.calls, "ledger_attempts": ledger.attempts(task.task_id),
                      "usage": llm.adapter.call_stats if llm.adapter else []}
            write_json(directory / "ready.json", marker)
            threading.Event().wait()
        llm = WorkerLLM(fake=args.fake, phase=args.phase, case=args.case, device=args.device,
                        max_tokens=args.max_new_tokens, stop=stop, model_path=args.model_path)
        def planner(state):
            attempts = []
            try:
                value = repair_plan(llm, prompt=prompt, catalog=catalog, inputs=task.inputs, question=task.question,
                    plan_prompt=PLAN_PROMPT, review_prompt=REVIEW_PROMPT, max_new_tokens=args.max_new_tokens,
                    max_repairs=2, attempts=attempts, ledger=ledger, task_id=state["task_id"],
                    policy_context={"model": args.model_path, "fake": args.fake, "case": args.case})
            except PlanRepairError:
                if args.phase == "prepare" and args.case == "exhausted":
                    stop("budget_exhausted_before_node_error")
                raise
            if args.phase == "prepare" and args.case == "accepted":
                stop("accepted_in_ledger_before_planner_checkpoint")
            return {"plan": value, "plan_attempts": attempts}
        graph = StateGraph(State).add_node("planner", planner).add_edge(START, "planner").add_edge("planner", END).compile(checkpointer=saver)
        config = {"configurable": {"thread_id": task.task_id}}
        before = graph.get_state(config)
        status, error, value = "complete", None, None
        try:
            value = graph.invoke({"task_id": task.task_id} if args.phase == "prepare" else None,
                                 config, durability="sync")
        except PlanRepairError as exc:
            status, error = "PlanRepairError", str(exc)
        after = graph.get_state(config)
        replay_calls = None
        if status == "complete":
            count = llm.calls
            graph.invoke(None, config, durability="sync")
            replay_calls = llm.calls - count
        # Directly replay the same node helper after completion/exhaustion. This
        # verifies the ledger bound independently of a finished graph's no-op.
        count = llm.calls
        try:
            repair_plan(llm, prompt=prompt, catalog=catalog, inputs=task.inputs, question=task.question,
                plan_prompt=PLAN_PROMPT, review_prompt=REVIEW_PROMPT, max_new_tokens=args.max_new_tokens,
                max_repairs=2, attempts=[], ledger=ledger, task_id=task.task_id,
                policy_context={"model": args.model_path, "fake": args.fake, "case": args.case})
        except PlanRepairError:
            pass
        report = {"phase": args.phase, "pid": os.getpid(), "case": args.case, "fake": args.fake,
                  "checkpoint_before_next": list(before.next), "checkpoint_after_next": list(after.next),
                  "calls": llm.calls, "ledger_attempts": ledger.attempts(task.task_id),
                  "status": status, "error": error, "value": value,
                  "completed_graph_reinvoke_calls": replay_calls, "direct_ledger_reinvoke_calls": llm.calls - count,
                  "usage": llm.adapter.call_stats if llm.adapter else [],
                  "elapsed_sec": time.perf_counter() - started}
        write_json(directory / f"{args.phase}.json", report)
    print(json.dumps({"phase": args.phase, "pid": os.getpid(), "report": str(directory / f"{args.phase}.json")}))


if __name__ == "__main__":
    main()
