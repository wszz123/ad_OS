"""Reference evaluation bridge to the parent repository's original HF methods.

Run in a fresh process. Validation is standalone; model execution explicitly
requires the parent LatentMAS research checkout. No research source is modified.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

from tools.moreagent_data import CONCISE_CONTRACT, agent_item, load_questions, parse_answer, response_messages, sha256, summarize


PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT.parent


def prepare_identity_normalization(model, *, chunk_rows=4096):
    """Same disabled-realignment identity and FP32 mean row norm, bounded scratch.

    The parent builder computes a dense solve before replacing its result by I
    when latent_space_realign=False. No active learned/dense alignment is bypassed.
    """
    import torch
    if model.args.latent_space_realign:
        raise ValueError("identity preparation cannot replace enabled latent realignment")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    weight = model.model.get_input_embeddings().weight.detach()
    with torch.no_grad():
        norms = [block.float().norm(dim=1) for block in weight.split(chunk_rows)]
        target_norm = torch.cat(norms).mean().detach()
        matrix = torch.eye(weight.shape[1], device=weight.device, dtype=torch.float32)
    model._latent_realign_matrices[id(model.model)] = (matrix, target_norm)
    return {"mode": "disabled_realign_identity_with_tiled_fp32_row_norm", "chunk_rows": chunk_rows,
            "embedding_rows": weight.shape[0], "hidden_size": weight.shape[1]}


def research_backend():
    if not (RESEARCH / "models.py").is_file() or not (RESEARCH / "methods" / "latent_mas.py").is_file():
        raise RuntimeError("model evaluation requires the parent LatentMAS research checkout")
    loaded = sys.modules.get("coop")
    if loaded and Path(loaded.__file__).resolve().parent != RESEARCH / "coop":
        raise RuntimeError("run this bridge in a fresh CLI process, not inside the prototype web server")
    sys.path.insert(0, str(RESEARCH))
    from models import ModelWrapper
    from methods.baseline import BaselineMethod
    from methods.text_mas import TextMASMethod
    from methods.latent_mas import LatentMASMethod
    return ModelWrapper, BaselineMethod, TextMASMethod, LatentMASMethod


def evaluate(args, questions, manifest):
    import torch
    ModelWrapper, Baseline, TextMAS, LatentMAS = research_backend()
    from utils import set_seed

    set_seed(args.seed)
    directory = Path(args.output_root) / ("moreagent-" + uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    public = [row["public"] for row in questions]
    public_bytes = "\n".join(json.dumps(row, ensure_ascii=False) for row in public).encode()
    (directory / "questions.public.jsonl").write_bytes(public_bytes + b"\n")
    labels = [{"question_id": row["public"]["question_id"], "gold": row["gold"],
               "topic": row["topic"], "source_type": row["source_type"]} for row in questions]
    (directory / "labels.evaluation-only.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2))
    report = {"experiment": "moreagent_original_hf_paired", "status": "loading", "directory": str(directory),
              "started_at": datetime.now(timezone.utc).isoformat(), "config": vars(args), "dataset": manifest,
              "public_input_sha256": sha256(public_bytes), "selected_questions": len(questions),
              "memory_enabled": False, "retrieval_enabled": False, "methods": {},
              "response_contract": CONCISE_CONTRACT if args.response_style == "concise" else "original research prompts",
              "limitations": [
                  "Presentation JSON converted by whitelist; answers and relationship annotations never passed to method inputs.",
                  "Reference parent HF methods, not the prototype's embedding-only or newly proposed hybrid reasoning backend.",
                  "Realignment disabled: identity/FP32 mean embedding norm prepared in chunks, skipping the parent's discarded dense solve; preparation time recorded separately.",
                  "Original 30 questions were development data; 20 authored questions lack independent medical review.",
                  "One fixed-order greedy run; not an unseen MedQA benchmark or a clinical validation.",
                  "Same output cap per text generation; TextMAS generates four times and LatentMAS uses three latent rollouts plus final text; not equal FLOPs.",
                  "If response_style=concise, the same 60-word/final-answer-first suffix is appended to every role prompt; not unchanged published prompts.",
                  "No cross-question memory, RAG, or oracle-based memory admission; candidate relationships are not observed reuse.",
                  "KV byte count is logical handoff tensor size; same-process references do not serialize those bytes onto a network."]}
    path = directory / "report.json"

    def persist():
        for cell in report["methods"].values():
            cell["summary"] = summarize(cell["results"])
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        temporary.replace(path)

    persist()
    method_args = argparse.Namespace(
        task="medqa", prompt="sequential", model_name=args.model_path, device=args.device, device2=args.device,
        method="latent_mas", max_new_tokens=args.max_new_tokens, disable_thinking=True, think=False,
        latent_only=False, sequential_info_only=False, latent_space_realign=False,
        latent_protocol_format="compact", text_mas_context_length=-1,
        use_vllm=False, enable_prefix_caching=False, use_second_HF_model=False)
    device = torch.device(args.device)
    started = time.perf_counter()
    try:
        model = ModelWrapper(args.model_path, device, use_vllm=False, args=method_args)
        report["model_load_sec"] = time.perf_counter() - started
        original_render = model.render_chat

        def render(messages, add_generation_prompt=True):
            return original_render(response_messages(messages, args.response_style), add_generation_prompt=add_generation_prompt)

        model.render_chat = render
        # Identical disabled-realign normalization, without its discarded solve.
        # Setup is recorded separately and is not charged to a single first item.
        if "latent_mas" in args.methods:
            started = time.perf_counter()
            report["latent_normalization_setup"] = prepare_identity_normalization(model)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            report["latent_prepare_sec"] = time.perf_counter() - started
        report["model_dtype"] = str(next(model.model.parameters()).dtype)
        report["status"] = "running"
        for name in args.methods:
            report["methods"][name] = {"status": "running", "results": []}
        persist()
        counter = {"forward": 0}
        hook = model.model.register_forward_hook(lambda *unused: counter.__setitem__("forward", counter["forward"] + 1))
        try:
            for row in questions:
                for name in args.methods:
                    report["active"] = {"method": name, "question_id": row["public"]["question_id"], "sequence": row["sequence"]}
                    persist()
                    current_args = argparse.Namespace(**{**vars(method_args), "method": name})
                    if name == "baseline":
                        method = Baseline(model, max_new_tokens=args.max_new_tokens, temperature=0, top_p=1,
                                          generate_bs=1, args=current_args)
                    elif name == "text_mas":
                        method = TextMAS(model, max_new_tokens_each=args.max_new_tokens, temperature=0, top_p=1,
                                         generate_bs=1, args=current_args)
                    else:
                        method = LatentMAS(model, latent_steps=args.latent_steps, judger_max_new_tokens=args.max_new_tokens,
                                           temperature=0, top_p=1, generate_bs=1, args=current_args)
                    generated = []
                    calls = []
                    original_generate = model.model.generate
                    original_text, original_latent = model.generate_text_batch, model.generate_latent_batch

                    def audited_generate(*positional, **keywords):
                        output = original_generate(*positional, **keywords)
                        sequences = output.sequences if hasattr(output, "sequences") else output
                        ids = keywords.get("input_ids", positional[0] if positional else None)
                        count = int(sequences.shape[-1] - ids.shape[-1])
                        generated.append({"tokens": count, "budget_saturated": count >= keywords.get("max_new_tokens", args.max_new_tokens)})
                        return output

                    def counted(kind, function):
                        def wrapped(*positional, **keywords):
                            calls.append(kind)
                            return function(*positional, **keywords)
                        return wrapped

                    model.model.generate = audited_generate
                    model.generate_text_batch = counted("text", original_text)
                    model.generate_latent_batch = counted("latent", original_latent)
                    counter["forward"] = 0
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    started = time.perf_counter()
                    inference_elapsed = None
                    result_row = {"question_id": row["public"]["question_id"], "sequence": row["sequence"],
                                  "topic": row["topic"], "source_type": row["source_type"], "gold": row["gold"]}
                    try:
                        # Gold is empty, even in method-side objects. Only this evaluator scores it.
                        result = method.run_batch([agent_item(row)])[0]
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        inference_elapsed = time.perf_counter() - started
                        prediction, parse_status = parse_answer(result["raw_prediction"])
                        agents = result["agents"]
                        protocol = method.latent_protocol.stats() if name == "latent_mas" else {}
                        result_row.update(
                            status="ok", prediction=prediction, parse_status=parse_status,
                            correct=prediction == row["gold"], raw_prediction=result["raw_prediction"],
                            legacy_prediction=result.get("prediction"),
                            prompt_tokens=sum(len(agent.get("input_ids", [])) for agent in agents),
                            generated_tokens=sum(call["tokens"] for call in generated),
                            intermediate_text_tokens=sum(len(model.tokenizer.encode(agent.get("output", ""), add_special_tokens=False))
                                                         for agent in agents[:-1]),
                            budget_saturated_calls=sum(call["budget_saturated"] for call in generated),
                            final_budget_saturated=bool(generated and generated[-1]["budget_saturated"]),
                            kv_tensor_bytes_logical=protocol.get("kv_tensor_payload_bytes_logical", 0),
                            protocol_frame_bytes=protocol.get("frame_utf8_bytes", 0),
                            latent_handoffs=protocol.get("handoff_count", 0))
                        (directory / f"{row['sequence']:02}-{name}-trace.json").write_text(
                            json.dumps({"agents": agents, "protocol": protocol}, ensure_ascii=False, indent=2))
                    except Exception as exc:
                        result_row.update(status="error", correct=False, error=f"{type(exc).__name__}: {exc}")
                    finally:
                        model.model.generate = original_generate
                        model.generate_text_batch, model.generate_latent_batch = original_text, original_latent
                    total_elapsed = time.perf_counter() - started
                    result_row.update(elapsed_sec=inference_elapsed if inference_elapsed is not None else total_elapsed,
                                      audit_sec=total_elapsed - inference_elapsed if inference_elapsed is not None else 0,
                                      forward_calls=counter["forward"],
                                      text_generation_calls=calls.count("text"), latent_rollout_calls=calls.count("latent"),
                                      latent_recurrent_steps=calls.count("latent") * args.latent_steps,
                                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
                    report["methods"][name]["results"].append(result_row)
                    persist()
                    print(f"{row['sequence']:02}/{len(questions)} {name} {row['public']['question_id']} "
                          f"pred={result_row.get('prediction')} gold={row['gold']} correct={result_row['correct']} "
                          f"time={result_row['elapsed_sec']:.2f}s", flush=True)
                    del method
                    if result_row["status"] != "ok":
                        raise RuntimeError(result_row["error"])
        finally:
            hook.remove()
        report["status"] = "complete"
        report.pop("active", None)
        for cell in report["methods"].values():
            cell["status"] = "complete"
        persist()
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        persist()
        raise
    print(json.dumps({"directory": str(directory), "status": report["status"],
                      "summaries": {name: cell["summary"] for name, cell in report["methods"].items()}}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(PROJECT / "moreagent_50_questions.json"))
    parser.add_argument("--model-path", default="/home/Users/SharedFile/Models/Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--methods", nargs="+", choices=("baseline", "text_mas", "latent_mas"), default=["baseline", "text_mas", "latent_mas"])
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--latent-steps", type=int, default=4)
    parser.add_argument("--response-style", choices=("original", "concise"), default="original")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default=str(PROJECT / "results" / "moreagent"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.max_samples < 1 or args.max_new_tokens < 1 or args.latent_steps < 0 or len(set(args.methods)) != len(args.methods):
        parser.error("positive sample/token budgets and unique methods required; latent steps must be nonnegative")
    questions, manifest = load_questions(args.dataset)
    if args.validate_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    evaluate(args, questions[:args.max_samples], manifest)


if __name__ == "__main__":
    main()
