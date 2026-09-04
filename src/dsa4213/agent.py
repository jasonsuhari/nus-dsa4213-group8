"""Chat agent with a pluggable memory approach."""

import json
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI

from .memory import EmbeddingMemory, KVMemory, Memory, Session, SummaryMemory, Turn

DEFAULT_MODEL = "gpt-4o-mini"

# Guard against a model that keeps calling the tool instead of answering.
MAX_TOOL_ROUNDS = 4

SYSTEM = """You are a personal assistant talking to one user over many sessions.

Use what you know about the user below. If you do not know something, say so
rather than guessing. When the user tells you something worth keeping for later,
call the remember tool.

What you know about the user:
{context}"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Save a fact about the user so it is available in future sessions. "
                "Call this when the user shares something worth keeping."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The fact, in one sentence."}
                },
                "required": ["fact"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


class Agent:
    """A chat agent. The memory approach is injected, everything else is fixed."""

    def __init__(
        self,
        memory: Memory,
        model: str = DEFAULT_MODEL,
        client: OpenAI | None = None,
    ) -> None:
        self.memory = memory
        self.model = model
        # Pass a client with a different base_url to run this against another provider.
        self.client = client or OpenAI()
        # Every fact the agent chose to save. Reconstruction is read off this.
        self.remembered: list[str] = []

    def ingest(self, session: Session) -> None:
        self.memory.write(session)

    def forget(self, target: str) -> None:
        self.memory.forget(target)

    def chat(self, message: str, date: str | None = None) -> str:
        messages = [
            {"role": "system", "content": SYSTEM.format(context=self.memory.retrieve(message))},
            {"role": "user", "content": message},
        ]
        for _ in range(MAX_TOOL_ROUNDS):
            reply = self._complete(messages)
            if not reply.tool_calls:
                return reply.content or ""
            messages.append(reply)
            for call in reply.tool_calls:
                fact = json.loads(call.function.arguments)["fact"]
                self._remember(fact, date or _today())
                messages.append({"role": "tool", "tool_call_id": call.id, "content": "saved"})
        return self._complete(messages).content or ""

    def _complete(self, messages: list):
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            tools=TOOLS,
            messages=messages,
        )
        return response.choices[0].message

    def _remember(self, fact: str, date: str) -> None:
        self.remembered.append(fact)
        session = Session(
            session_id=f"remember-{len(self.remembered):03d}",
            date=date,
            turns=(Turn("assistant", fact),),
        )
        self.memory.write(session)


def EmbeddingRetrievalAgent(path: Path, **kwargs) -> Agent:
    return Agent(EmbeddingMemory(path), **kwargs)


def SummaryAgent(path: Path, **kwargs) -> Agent:
    return Agent(SummaryMemory(path), **kwargs)


def KVAgent(path: Path, **kwargs) -> Agent:
    return Agent(KVMemory(path), **kwargs)
