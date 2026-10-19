"""Dependency-light local API for the competition demonstration dashboard."""

from dataclasses import asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import threading
import time
from uuid import uuid4

from .demo_support import EMBEDDING, QWEN, FixturePlanner, fixture_encode
from .memory import MemoryAdapter
from .protocol import ProtocolAdapter
from .text_transport import TextAdapter
from .workload import cost_documents, cost_tasks, variable_catalog


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
PRESETS = {task.task_id: task for task in cost_tasks()}
RUNS = {}
RUNS_LOCK = threading.RLock()
ENGINE = {}
ENGINE_LOCK = threading.RLock()
EXECUTION_LOCK = threading.Lock()
ROLES = [
    {"id": "planner", "label": "Planner", "role": "任务规划"},
    {"id": "retriever", "label": "Retriever", "role": "资料检索"},
    {"id": "executor", "label": "Executor", "role": "工具执行"},
    {"id": "summarizer", "label": "Summarizer", "role": "总结生成"},
]


def system_status():
    return {"roles": ROLES, "mode": ENGINE.get("backend", "fixture"),
            "model": ENGINE.get("model_path", QWEN) if ENGINE.get("backend") == "qwen" else "规则演示器（非LLM）",
            "embedding": ENGINE.get("embedding_path", EMBEDDING) if ENGINE.get("backend") == "qwen" else "2维主题夹具（非学习向量）",
            "loaded": bool(ENGINE.get("llm")) if ENGINE.get("backend") == "qwen" else True,
            "last_error": ENGINE.get("error")}


def start_engine(backend="fixture", device="cuda:0", embedding_device="cuda:1",
                 *, model_path=QWEN, embedding_path=EMBEDDING):
    if backend not in ("fixture", "qwen"):
        raise ValueError("backend must be fixture or qwen")
    with ENGINE_LOCK:
        if (ENGINE.get("backend") == backend and ENGINE.get("device") == device
                and ENGINE.get("embedding_device") == embedding_device
                and ENGINE.get("model_path") == model_path and ENGINE.get("embedding_path") == embedding_path
                and (backend == "fixture" or ENGINE.get("llm"))):
            return system_status()
        for name in ("memory", "corpus"):
            old = ENGINE.get(name)
            if old is not None:
                old.close()
        ENGINE.clear()
        ENGINE.update(backend=backend, device=device, embedding_device=embedding_device,
                      model_path=model_path, embedding_path=embedding_path, error=None)
        ENGINE["memory"] = MemoryAdapter(check_same_thread=False)
        ENGINE["corpus"] = None
        if backend == "fixture":
            ENGINE.update(llm=FixturePlanner(), encoder=fixture_encode)
        else:
            from .embedding import LocalQwenEmbedding
            from .qwen_agent import QwenCausalAdapter

            llm = QwenCausalAdapter(model_path, device=device, max_new_tokens=192)
            embedding = LocalQwenEmbedding(embedding_path, device=embedding_device, output_dim=768)
            ENGINE.update(llm=llm, embedding_model=embedding, encoder=embedding.encode)
            try:
                llm.load()
                embedding.encode(["dashboard warmup"], query=True)
            except Exception as exc:
                ENGINE["error"] = f"{type(exc).__name__}: {exc}"
                raise
    return system_status()


def prepare_corpus(entity_filter=None):
    with ENGINE_LOCK:
        if ENGINE.get("corpus") is None:
            corpus = MemoryAdapter(check_same_thread=False)
            docs = cost_documents()
            vectors = ENGINE["encoder"]([item.summary for item in docs])
            for item, vector in zip(docs, vectors):
                item.embedding = vector.tolist() if hasattr(vector, "tolist") else list(vector)
                corpus.put(item)
            ENGINE["corpus"] = corpus
    corpus = ENGINE["corpus"]
    if entity_filter:
        return _FilteredMemoryBackend(corpus, entity_filter), ENGINE["memory"]
    return corpus, ENGINE["memory"]


class _FilteredMemoryBackend:
    """Entity-scoped view that leaves the shared source corpus immutable."""

    def __init__(self, backend, entity):
        self.backend, self.entity = backend, entity

    def put(self, item): return self.backend.put(item)
    def get(self, memory_id):
        item = self.backend.get(memory_id)
        return item if item is not None and item.task_topic == self.entity else None
    def get_metadata(self, memory_id):
        item = self.backend.get_metadata(memory_id)
        return item if item is not None and item.task_topic == self.entity else None
    def items(self): return [item for item in self.backend.items() if item.task_topic == self.entity]
    def count(self): return len(self.items())
    def close(self): pass
    def search(self, query, *, tags=None, limit=5):
        return self.backend.search(query, tags=[*(tags or []), "entity:" + self.entity], limit=limit)
    def search_vector(self, vector, *, tags=None, limit=5):
        return self.backend.search_vector(vector, tags=[*(tags or []), "entity:" + self.entity], limit=limit)


def normalize_user_task(query, entity, inputs):
    if entity not in ("Orion", "Lyra"):
        raise ValueError("entity must be Orion or Lyra")
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        raise ValueError("task must be between 1 and 2000 characters")
    if not isinstance(inputs, dict) or len(inputs) > 16:
        raise ValueError("inputs must be a JSON object with at most 16 fields")
    catalog = variable_catalog(entity)
    allowed = set(catalog) | {"node_count", "days", "months", "retention_days", "replicas", "egress_fraction"}
    if set(inputs) - allowed:
        raise ValueError(f"unsupported input fields: {sorted(set(inputs) - allowed)}")
    for name, value in inputs.items():
        if len(str(value)) > 32:
            raise ValueError(f"input {name} is too long")
    return {name: str(value) for name, value in inputs.items()}


def execute_one(*, query, entity, inputs, mode, use_vector, use_memory, run_id, task_id=None):
    if mode not in ("text", "protocol"):
        raise ValueError("mode must be text or protocol")
    clean_inputs = normalize_user_task(query, entity, inputs)
    llm, encoder, task_memory = ENGINE["llm"], ENGINE["encoder"], ENGINE["memory"]
    from .cost_graph import build_cost_graph
    from .answer_validation import validate_cost_answer
    corpus, _ = prepare_corpus(entity)
    protocol = TextAdapter() if mode == "text" else ProtocolAdapter()
    state_packets = {}
    graph, _, events = build_cost_graph(
        llm, corpus, task_memory, protocol, namespace=f"web:{ENGINE.get('model_path', ENGINE.get('backend'))}:{entity}",
        encoder=encoder if use_vector else None, reuse=use_memory,
        max_new_tokens=192, state_store=state_packets,
    )
    task_id = task_id or f"web-{run_id}"
    before = len(getattr(llm, "call_stats", []))
    started = time.perf_counter()
    result = graph.invoke({"task_id": task_id, "entity": entity, "question": query,
                           "inputs_json": json.dumps(clean_inputs, sort_keys=True), "unit": "CNY"})
    elapsed = time.perf_counter() - started
    calls = getattr(llm, "call_stats", [])[before:]
    answer = json.loads(result["answer_json"])
    tool = json.loads(result["tool_json"])
    sources = json.loads(result["documents_json"])
    frames = protocol.frames
    vector = state_packets.get(task_id + ":facts-query")
    # This window is a manual quality indicator, not a ground-truth grade:
    # required citations come from this task's actually retrieved sources.
    task_for_validation = type("RetrievedTask", (), {
        "expected_value": tool["value"], "required_citations": tuple(tool["citations"]), "unit": tool["unit"]
    })()
    quality = validate_cost_answer(task_for_validation, answer, tool)
    task_metrics = events.get(task_id, {})
    return {
        "run_id": run_id, "task_id": task_id, "mode": mode,
        "query": query, "entity": entity, "inputs": clean_inputs,
        "answer": answer, "tool_result": tool,
        "evidence": sources, "evidence_ids": [doc["memory_id"] for doc in sources],
        "reused_memory_id": result.get("reused_memory_id", ""),
        "quality": quality,
        "validation_scope": "internal calculator/source consistency; no hidden-label grade for manual tasks",
        "elapsed_sec": round(elapsed, 4),
        "llm_calls": len(calls), "prompt_tokens": sum(call["prompt_tokens"] for call in calls),
        "generated_tokens": sum(call["generated_tokens"] for call in calls),
        "message_count": len(frames), "message_chars": sum(len(frame) for frame in frames),
        "message_bytes": sum(len(frame.encode("utf-8")) for frame in frames),
        "state_transfer_count": 1 if vector else 0,
        "state_payload_bytes": vector.payload_bytes if vector else 0,
        "state_encoding": vector.encoding if vector else None,
        "accepted_reuse": task_metrics.get("accepted_reuse", 0),
        "reuse_source_task": task_metrics.get("reuse_source_task"),
        "corpus_searches": task_metrics.get("corpus_searches", 0),
        "roles": [{**role, "status": "done"} for role in ROLES],
        "trace": frames,
        "capability_discovery": [asdict(capability) for capability in protocol.discover("plan")],
    }


def enqueue_run(payload):
    run_id = uuid4().hex[:12]
    record = {"run_id": run_id, "status": "queued", "created_at": datetime.now(timezone.utc).isoformat(),
              "payload": payload, "result": None, "error": None}
    with RUNS_LOCK:
        RUNS[run_id] = record

    def execute():
        with RUNS_LOCK:
            RUNS[run_id]["status"] = "running"
        try:
            result = execute_one(run_id=run_id, **payload)
            with RUNS_LOCK:
                RUNS[run_id].update(status="complete", result=result,
                                    completed_at=datetime.now(timezone.utc).isoformat())
        except Exception as exc:
            with RUNS_LOCK:
                RUNS[run_id].update(status="error", error=f"{type(exc).__name__}: {exc}",
                                    completed_at=datetime.now(timezone.utc).isoformat())

    threading.Thread(target=execute, name=f"ui-run-{run_id}", daemon=True).start()
    return record


class Handler(BaseHTTPRequestHandler):
    server_version = "CoopPrototype/1.0"

    def log_message(self, fmt, *args):
        print(f"[web] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > 32_000:
            raise ValueError("request body must be between 1 and 32000 bytes")
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        if self.path == "/api/health":
            return self._json(200, {"ok": True, **system_status()})
        if self.path == "/api/metrics":
            with RUNS_LOCK:
                completed = [item["result"] for item in RUNS.values()
                             if item["status"] == "complete" and item.get("result")]
            return self._json(200, {
                "run_count": len(RUNS), "completed_count": len(completed),
                "accepted_reuse": sum(item.get("accepted_reuse", 0) for item in completed),
                "state_transfers": sum(item.get("state_transfer_count", 0) for item in completed),
                "state_payload_bytes": sum(item.get("state_payload_bytes", 0) for item in completed),
                "messages": sum(item.get("message_count", 0) for item in completed),
                "message_bytes": sum(item.get("message_bytes", 0) for item in completed),
                "llm_calls": sum(item.get("llm_calls", 0) for item in completed),
                "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in completed),
                "generated_tokens": sum(item.get("generated_tokens", 0) for item in completed),
            })
        if self.path == "/api/presets":
            presets = [{"task_id": task.task_id, "entity": task.entity, "question": task.question,
                        "inputs": task.inputs, "required_citations": list(task.required_citations)}
                       for task in PRESETS.values()]
            return self._json(200, {"presets": presets})
        if self.path.startswith("/api/runs/"):
            run_id = self.path.removeprefix("/api/runs/")
            with RUNS_LOCK:
                run = RUNS.get(run_id)
            return self._json(200 if run else 404, run or {"error": "run not found"})
        if self.path == "/" or self.path.startswith("/index.html"):
            return self._static("index.html")
        if self.path.startswith("/assets/"):
            name = self.path.removeprefix("/assets/")
            if Path(name).name != name:
                return self._json(404, {"error": "not found"})
            return self._static(f"assets/{name}")
        return self._json(404, {"error": "not found"})

    def _static(self, name):
        path = WEB_ROOT / name
        if not path.is_file():
            return self._json(404, {"error": "static asset not found"})
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            payload = self._body()
            if self.path == "/api/engine":
                return self._json(200, start_engine(
                    payload.get("backend", "fixture"), payload.get("device", "cuda:0"),
                    payload.get("embedding_device", "cuda:1"),
                    model_path=payload.get("model_path", QWEN),
                    embedding_path=payload.get("embedding_path", EMBEDDING)))
            if self.path == "/api/run":
                preset_id = payload.get("task_id")
                if preset_id:
                    task = PRESETS.get(preset_id)
                    if task is None:
                        return self._json(404, {"error": "unknown preset task"})
                    payload.update(query=task.question, entity=task.entity, inputs=task.inputs)
                required = ("query", "entity", "inputs", "mode")
                if any(key not in payload for key in required):
                    return self._json(400, {"error": f"required fields: {', '.join(required)}"})
                normalize_user_task(payload["query"], payload["entity"], payload["inputs"])
                if payload.get("mode") not in ("text", "protocol"):
                    return self._json(400, {"error": "mode must be text or protocol"})
                if not ENGINE.get("backend"):
                    start_engine(payload.get("backend", "fixture"))
                run_id = uuid4().hex[:12]
                record = {"run_id": run_id, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
                          "payload": {key: payload[key] for key in required}, "result": None, "error": None}
                with RUNS_LOCK:
                    RUNS[run_id] = record
                # Sequential execution avoids racing the shared deterministic LLM/cache,
                # matching this local single-user demonstration application's intended use.
                with EXECUTION_LOCK:
                    result = execute_one(run_id=run_id, use_vector=bool(payload.get("use_vector", True)),
                                         use_memory=bool(payload.get("memory", True)),
                                         task_id=payload.get("task_id"),
                                         **{key: payload[key] for key in required if key != "mode"},
                                         mode=payload["mode"])
                record.update(status="complete", result=result, completed_at=datetime.now(timezone.utc).isoformat())
                return self._json(200, record)
        except Exception as exc:
            return self._json(400, {"error": f"{type(exc).__name__}: {exc}"})
        return self._json(404, {"error": "endpoint not found"})


def serve(host="127.0.0.1", port=8765):
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Coop demo UI: http://{host}:{port}", flush=True)
    print(f"Runtime: {ENGINE.get('backend', 'fixture')}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
