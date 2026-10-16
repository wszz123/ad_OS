"""Validated action envelopes, capabilities and discovery for graph nodes."""

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Callable, Dict, List
from uuid import uuid4


@dataclass(frozen=True)
class Capability:
    agent_id: str
    role: str
    actions: tuple
    version: str = "1.0"


@dataclass
class Message:
    source: str
    target: str
    action: str
    params: Dict[str, Any]
    message_id: str = field(default_factory=lambda: str(uuid4()))
    version: str = "1.0"


class ProtocolAdapter:
    """Local dispatch boundary; all parameters/results are serialized and counted."""

    def __init__(self) -> None:
        self.capabilities: Dict[str, Capability] = {}
        self.handlers: Dict[str, Dict[str, Callable]] = {}
        self.frames: List[str] = []

    def register(self, capability: Capability, handlers: Dict[str, Callable]) -> None:
        if capability.version != "1.0":
            raise ValueError("unsupported protocol version")
        if not capability.agent_id or capability.agent_id in self.capabilities:
            raise ValueError("empty or duplicate agent id")
        if set(handlers) != set(capability.actions):
            raise ValueError("capabilities and handlers must agree")
        self.capabilities[capability.agent_id] = capability
        self.handlers[capability.agent_id] = handlers

    def discover(self, action: str) -> List[Capability]:
        return [item for item in self.capabilities.values() if action in item.actions]

    def dispatch(self, message: Message) -> Dict[str, Any]:
        if message.version != "1.0":
            raise ValueError("unsupported protocol version")
        if message.source not in self.capabilities or message.target not in self.capabilities:
            raise ValueError("endpoints must be registered")
        if message.action not in self.handlers[message.target]:
            raise ValueError("action not advertised by target")
        if not isinstance(message.params, dict):
            raise ValueError("parameters must be a dictionary")
        request = json.dumps(asdict(message), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        self.frames.append(request)
        result = self.handlers[message.target][message.action](json.loads(request)["params"])
        if not isinstance(result, dict):
            raise ValueError("result must be a dictionary")
        response = json.dumps({
            "source": message.target, "target": message.source, "action": message.action,
            "in_reply_to": message.message_id, "version": "1.0", "result": result,
        }, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        self.frames.append(response)
        return json.loads(response)["result"]
