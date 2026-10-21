"""Native child-repository medical MAS evaluation; no parent LatentMAS imports."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from uuid import uuid4

from coop.medical_graph import (MODES, FixtureMedicalDecision, QwenMedicalDecision,
                                build_medical_graph, create_medical_corpus, fixture_encode)
from coop.medical_memory import MedicalMemory
from coop.memory import MemoryAdapter
from coop.protocol import ProtocolAdapter
from coop.text_transport import TextAdapter
from tools.moreagent_data import load_questions, sha256, summarize


ROOT = Path(__file__).resolve().parents[1]


class EncoderMeter:
    def __init__(self, encoder):
        self.encoder = encoder
        self.calls = self.items = 0
        self.elapsed_sec = 0.0

    def __call__(self, texts, *, query=False):
        started = time.perf_counter()
        values = self.encoder(texts, query=query)
        self.calls += 1
        self.items += len(texts)
        self.elapsed_sec += time.perf_counter() - started
        return values


def medical_summary(rows):
    result = summarize(rows)
    for key in ("model_calls", "budget_saturated_calls", "message_count", "message_chars", "message_bytes", "state_transfers", "state_payload_bytes",
                "state_metadata_bytes", "total_transfer_bytes", "memory_queries", "memory_candidate_count",
                "retrieval_hits", "declared_reuse", "declared_cross_topic_reuse", "correct_with_declared_reuse",
                "memory_writes", "schema_retries", "role_calls", "embedding_sec"):
        result[key] = sum(row.get(key, 0) for row in rows)
    result["retrieval_hit_rate"] = result["retrieval_hits"] / result["memory_queries"] if result["memory_queries"] else None
    return result


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def run_mode(rows, directory, *, mode, backend="fixture", decision=None, encoder=fixture_encode,
             namespace=None, top_k=2, schema_attempts=2, on_update=None):
    if mode not in MODES:
        raise ValueError("unknown medical mode")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("medical mode directory must be empty")
    decision = decision or FixtureMedicalDecision()
    vector = "vector" in mode
    scope = "shared" if mode == "protocol_vector_memory" else "group" if mode == "protocol_vector_group_memory" else "none"
    meter = EncoderMeter(encoder) if vector else None
    backend_memory = MemoryAdapter(str(directory / "memory.sqlite"))
    memory = MedicalMemory(backend_memory, namespace=namespace or uuid4().hex,
                           model_id=decision.model_id, encoder=meter if scope != "none" else None)
    report = {"experiment": "native_medical_mas_mode", "implementation": "coop.medical_graph (child repository)",
              "backend": backend, "mode": mode, "selected_questions": len(rows), "status": "initializing",
              "initial_memory_count": backend_memory.count(), "results": [], "summary": {},
              "memory_policy": "source-quoted but clinically unverified model notes; no gold-based admission",
              "latent_kv_enabled": False, "state_kind": "embedding_query_float16" if vector else "none"}
    corpus = None
    forwards = {"count": 0}
    model = getattr(getattr(decision, "llm", None), "model", None)
    hook = (model.register_forward_hook(lambda *args: forwards.__setitem__("count", forwards["count"] + 1))
            if model is not None else None)
    report["forward_measurement"] = "HF causal-model forward hook" if hook is not None else "fixture has no model forwards"
    report["token_measurement"] = "actual JSON generation token counts including EOS; workflow/embedding forwards separate"

    def persist():
        report["summary"] = medical_summary(report["results"])
        write_json(directory / "report.json", report)
        if on_update:
            on_update(report)

    try:
        persist()
        started = time.perf_counter()
        corpus = create_medical_corpus(meter)
        report["corpus_setup_sec"] = time.perf_counter() - started
        report["corpus_scope"] = "five generic workflow checklists; no medical answers or future cases"
        report["status"] = "running"
        for item in rows:
            public, sequence = item["public"], item["sequence"]
            report["active"] = {"sequence": sequence, "question_id": public["question_id"]}
            persist()
            transport = TextAdapter() if mode == "text" else ProtocolAdapter()
            graph, packets, metrics, _ = build_medical_graph(
                decision, memory, transport, encoder=meter, use_vector=vector,
                memory_scope=scope, top_k=top_k, schema_attempts=schema_attempts, corpus=corpus)
            before_calls = len(decision.call_stats)
            before_prompts = len(decision.prompt_records)
            before_forwards = forwards["count"]
            before_embedding = meter.elapsed_sec if meter else 0.0
            started = time.perf_counter()
            row = {"question_id": public["question_id"], "sequence": sequence}
            try:
                state = graph.invoke({"public": public, "sequence": sequence})
                final = state["final"]
                # Publish BEFORE evaluating: no evaluator labels or correctness gate reach memory.
                admission = (memory.publish(public, sequence, state["plan"], final) if scope != "none"
                             else {"written": False, "reason": "memory_disabled"})
                elapsed = time.perf_counter() - started
                used = final["used_memory_ids"]
                selected = state["retrieval"]["memories"]
                used_set = set(used)
                row.update(status="ok", prediction=final["answer"], parse_status="json_choice",
                           raw_prediction=json.dumps(final, ensure_ascii=False), final=final, plan=state["plan"], draft=state["draft"],
                           retrieval=state["retrieval"], admission=admission,
                           retrieval_hits=int(bool(selected)), declared_reuse=len(used),
                           declared_cross_topic_reuse=sum(m["memory_id"] in used_set and m["topic"] != state["plan"]["topic"] for m in selected),
                           memory_writes=int(admission["written"]), elapsed_sec=elapsed)
            except Exception as exc:
                row.update(status="error", prediction=None, parse_status="missing", declared_reuse=0,
                           memory_writes=0, elapsed_sec=time.perf_counter() - started,
                           error=f"{type(exc).__name__}: {exc}")
            # Evaluator-only fields enter the result after graph and admission are finished.
            row.update(gold=item["gold"], topic=item["topic"], source_type=item["source_type"],
                       correct=row["prediction"] == item["gold"] and row["status"] == "ok")
            row["correct_with_declared_reuse"] = int(row["correct"] and row["declared_reuse"] > 0)
            row.update(metrics)
            calls = decision.call_stats[before_calls:]
            prompt_records = decision.prompt_records[before_prompts:]
            row.update(model_calls=len(calls), prompt_tokens=sum(call["prompt_tokens"] for call in calls),
                       generated_tokens=sum(call["generated_tokens"] for call in calls),
                       intermediate_text_tokens=sum(record.get("usage", {}).get("generated_tokens", 0)
                                                    for record in prompt_records if record["role"] != "verifier"),
                       forward_calls=forwards["count"] - before_forwards, text_generation_calls=len(calls), latent_rollout_calls=0,
                       embedding_sec=(meter.elapsed_sec - before_embedding) if meter else 0.0,
                       message_count=len(transport.frames), message_chars=sum(len(frame) for frame in transport.frames),
                       message_bytes=sum(len(frame.encode()) for frame in transport.frames))
            row["budget_saturated_calls"] = sum(record.get("usage", {}).get("generated_tokens", 0) >= record.get("max_new_tokens", 1)
                                                 for record in prompt_records)
            row["total_transfer_bytes"] = row["message_bytes"] + row["state_payload_bytes"] + row["state_metadata_bytes"]
            write_json(directory / f"{sequence:02}-trace.json", {"frames": transport.frames,
                       "capabilities": {agent: asdict(value) for agent, value in transport.capabilities.items()},
                       "model_prompts": prompt_records})
            for packet in packets.values():
                metadata = asdict(packet)
                metadata.pop("payload")
                (directory / f"{sequence:02}-query.bin").write_bytes(packet.payload)
                write_json(directory / f"{sequence:02}-query.json", metadata)
            report["results"].append(row)
            persist()
            print(f"{mode} {sequence:02}/{len(rows)} {public['question_id']} pred={row['prediction']} "
                  f"gold={item['gold']} correct={row['correct']} memory_write={row['memory_writes']}", flush=True)
        report.pop("active", None)
        report["status"] = "complete" if all(r["status"] == "ok" for r in report["results"]) else "complete_with_errors"
        report["final_memory_count"] = backend_memory.count()
        persist()
        return report
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        persist()
        raise
    finally:
        backend_memory.close()
        if hook is not None:
            hook.remove()
        if corpus is not None:
            corpus.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(ROOT / "moreagent_50_questions.json"))
    parser.add_argument("--backend", choices=("fixture", "qwen"), default="fixture")
    parser.add_argument("--modes", nargs="+", choices=(*MODES, "all"), default=["all"])
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--schema-attempts", type=int, default=2)
    parser.add_argument("--model-path", default="/home/Users/SharedFile/Models/Qwen/Qwen3-8B")
    parser.add_argument("--embedding-path", default="/home/Users/SharedFile/Models/Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--output-root", default=str(ROOT / "results" / "medical-native"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.max_samples < 1 or not 1 <= args.top_k <= 5 or not 1 <= args.schema_attempts <= 3:
        parser.error("invalid sample, retrieval or schema-retry budget")
    if len(set(args.modes)) != len(args.modes) or "all" in args.modes and len(args.modes) != 1:
        parser.error("use unique modes or all by itself")
    rows, manifest = load_questions(args.dataset)
    if args.validate_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    rows = rows[:args.max_samples]
    modes = MODES if args.modes == ["all"] else tuple(args.modes)
    directory = Path(args.output_root) / ("medical-" + args.backend + "-" + uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    public_bytes = "\n".join(json.dumps(row["public"], ensure_ascii=False) for row in rows).encode()
    (directory / "questions.public.jsonl").write_bytes(public_bytes + b"\n")
    write_json(directory / "labels.evaluation-only.json", [{"question_id": row["public"]["question_id"],
               "gold": row["gold"], "topic": row["topic"], "source_type": row["source_type"]} for row in rows])
    report = {"experiment": "native_medical_mas_paired", "status": "loading", "config": vars(args),
              "dataset": manifest, "selected_questions": len(rows), "public_input_sha256": sha256(public_bytes),
              "full_dataset": len(rows) == manifest["total_questions"],
              "implementation": "child repository only; no parent methods/models imports", "modes": {},
              "token_budgets": {"planner": 256, "executor": 256, "verifier": 512},
              "limitations": [
                  "Constructed medical-related dataset: 30 development-used MedQA and 20 unreviewed authored questions.",
                  "Generic workflow checklists are not a medical evidence database; no oracle hints or relation annotations are loaded.",
                  "Case quotations verify source transcription, NOT clinical truth. Model notes remain unverified even if agents agree.",
                  "Memory is empty per mode, online in supplied order; gold and correctness never decide admission.",
                  "Group/topic filters come from the planner's case-derived topic, not evaluator topic labels.",
                  "Declared reuse and nonempty condition checks are model claims, not independently verified medical applicability.",
                  "Correct-with-declared-reuse is descriptive and does not establish causal benefit.",
                  "Text/Protocol deliver identical current semantic fields; text serializes actual within-task history. Only transport bytes differ by design.",
                  "Native three-model-call/four-role JSON pipeline is not the parent's four-generating-role TextMAS or latent KV method.",
                  "Query embeddings are used for workflow and memory retrieval; no latent KV continuation or remote wire benchmark."]}
    write_json(directory / "report.json", report)
    try:
        decision, encoder = FixtureMedicalDecision(), fixture_encode
        if args.backend == "qwen":
            from coop.qwen_agent import QwenCausalAdapter
            from coop.embedding import LocalQwenEmbedding
            llm = QwenCausalAdapter(args.model_path, device=args.device, max_new_tokens=512)
            started = time.perf_counter()
            llm.load()
            report["model_load_sec"] = time.perf_counter() - started
            report["model_dtype"] = str(next(llm.model.parameters()).dtype)
            decision = QwenMedicalDecision(llm)
            encoder = LocalQwenEmbedding(args.embedding_path, device=args.embedding_device, output_dim=768).encode
        report["status"] = "running"
        for mode in modes:
            def update(value, mode=mode):
                report["modes"][mode] = value
                write_json(directory / "report.json", report)
            run_mode(rows, directory / mode, mode=mode, backend=args.backend, decision=decision,
                     encoder=encoder, namespace=directory.name + ":" + mode, top_k=args.top_k,
                     schema_attempts=args.schema_attempts, on_update=update)
        report["status"] = "complete" if all(value["status"] == "complete" for value in report["modes"].values()) else "complete_with_errors"
        write_json(directory / "report.json", report)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(directory / "report.json", report)
        raise
    print(json.dumps({"directory": str(directory), "status": report["status"],
                      "summaries": {mode: value["summary"] for mode, value in report["modes"].items()}}, indent=2, ensure_ascii=False))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
