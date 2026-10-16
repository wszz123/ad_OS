"""Lossless text handoffs with actual per-task conversation history."""

from typing import Dict

from .protocol import Message, ProtocolAdapter


def render_fields(fields: Dict[str, str]) -> str:
    """Human-readable length-delimited blocks preserve arbitrary multiline values."""
    blocks = []
    for key, value in fields.items():
        if not isinstance(value, str) or not key.isidentifier():
            raise ValueError("text handoffs require identifier keys and string values")
        blocks.append(f"Field {key} contains {len(value)} characters:\n{value}\n")
    return "".join(blocks)


def parse_fields(raw: str) -> Dict[str, str]:
    fields = {}
    while raw:
        header, separator, body = raw.partition("\n")
        parts = header.split()
        if not separator or len(parts) != 5 or parts[0] != "Field" or parts[2] != "contains" or parts[4] != "characters:":
            raise ValueError("invalid text field header")
        length = int(parts[3])
        if length < 0 or len(body) < length + 1 or body[length] != "\n":
            raise ValueError("invalid text field length")
        key = parts[1]
        if not key.isidentifier() or key in fields:
            raise ValueError("invalid or duplicate text field")
        fields[key] = body[:length]
        raw = body[length + 1:]
    return fields


class TextAdapter(ProtocolAdapter):
    """Same registry/handlers, text-only transport instead of JSON envelopes.

    Context is the actual earlier request/response bodies within this task. It
    resets between tasks; no artificial padding or fabricated conversation.
    """

    def __init__(self):
        super().__init__()
        self.history = []

    def begin_task(self):
        self.history.clear()

    def dispatch(self, message: Message) -> Dict:
        if message.version != "1.0":
            raise ValueError("unsupported protocol version")
        if message.source not in self.capabilities or message.target not in self.capabilities:
            raise ValueError("endpoints must be registered")
        if message.action not in self.handlers[message.target]:
            raise ValueError("action not advertised by target")
        body = render_fields(message.params)
        request = (f"{message.source} asks {message.target} to {message.action}.\n"
                   + "Prior task handoffs:\n" + "\n".join(self.history) + "\nCurrent fields:\n" + body)
        # Parsing operates on the transmitted body, not the original Python values.
        received = request[-len(body):] if body else ""
        self.frames.append(request)
        result = self.handlers[message.target][message.action](parse_fields(received))
        response_body = render_fields(result)
        response = f"{message.target} replies to {message.source}.\n" + response_body
        self.frames.append(response)
        self.history.extend((f"{message.action} request:\n{body}", f"{message.action} reply:\n{response_body}"))
        return parse_fields(response_body)
