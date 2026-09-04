"""Chat agent with a pluggable memory approach."""

from pathlib import Path

from openai import OpenAI

from .memory import EmbeddingMemory, KVMemory, Memory, Session, SummaryMemory

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM = """You are a personal assistant talking to one user over many sessions.

Use what you know about the user below. If you do not know something, say so
rather than guessing.

What you know about the user:
{context}"""


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

    def ingest(self, session: Session) -> None:
        self.memory.write(session)

    def forget(self, target: str) -> None:
        self.memory.forget(target)

    def chat(self, message: str) -> str:
        context = self.memory.retrieve(message)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM.format(context=context)},
                {"role": "user", "content": message},
            ],
        )
        return response.choices[0].message.content


def EmbeddingRetrievalAgent(path: Path, **kwargs) -> Agent:
    return Agent(EmbeddingMemory(path), **kwargs)


def SummaryAgent(path: Path, **kwargs) -> Agent:
    return Agent(SummaryMemory(path), **kwargs)


def KVAgent(path: Path, **kwargs) -> Agent:
    return Agent(KVMemory(path), **kwargs)
