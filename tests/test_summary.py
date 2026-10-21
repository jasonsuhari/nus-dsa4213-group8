from src.dsa4213.memory.summary import SummaryMemory
from unittest.mock import MagicMock
from src.dsa4213.memory.base import Session, Turn


def test_empty_summary(tmp_path):
    memory = SummaryMemory(path=tmp_path)

    assert memory.dump() == ""
    assert memory.retrieve("What do you know about me?") == ""


def test_summary_persistence(tmp_path):
    memory = SummaryMemory(path=tmp_path)

    memory.summary = "The user studies at NUS."
    memory.save_summary()

    # Create a new instance using the same directory
    reloaded_memory = SummaryMemory(path=tmp_path)

    assert reloaded_memory.dump() == "The user studies at NUS."


def test_retrieve_max_chars(tmp_path):
    memory = SummaryMemory(path=tmp_path, max_chars=20)

    memory.summary = "The user studies at NUS and likes Japanese food."

    assert memory.retrieve("Where do I study?") == memory.summary[:20]
    assert len(memory.retrieve("Where do I study?")) <= 20
    assert memory.dump() == memory.summary


def test_write(tmp_path):
    # Create a mock OpenAI client
    mock_client = MagicMock()

    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(
            message=MagicMock(
                content="The user is vegetarian and studies at NUS."
            )
        )
    ]

    memory = SummaryMemory(path=tmp_path, client=mock_client)

    session = Session(
        session_id="session_1",
        date="2026-10-20",
        turns=(
            Turn(role="user", text="I am vegetarian."),
            Turn(role="user", text="I study at NUS."),
        )
    )

    memory.write(session)

    assert memory.dump() == "The user is vegetarian and studies at NUS."

    # Check that the updated summary was saved
    assert (tmp_path / "summary.txt").read_text(
        encoding="utf-8"
    ) == memory.dump()

    # Check that the LLM was called once
    mock_client.chat.completions.create.assert_called_once()


def test_forget(tmp_path):
    mock_client = MagicMock()

    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(
            message=MagicMock(
                content="The user likes Japanese food and studies at NUS."
            )
        )
    ]

    memory = SummaryMemory(path=tmp_path, client=mock_client)

    # Set up existing memory
    memory.summary = (
        "The user is vegetarian, likes Japanese food, "
        "and studies at NUS."
    )
    memory.save_summary()

    # Forget the targeted fact
    memory.forget("The user is vegetarian.")

    # Check that the summary was updated
    assert memory.dump() == (
        "The user likes Japanese food and studies at NUS."
    )

    # Check that the updated summary was saved
    assert (tmp_path / "summary.txt").read_text(
        encoding="utf-8"
    ) == memory.dump()

    # Check that the LLM was called once
    mock_client.chat.completions.create.assert_called_once()


def test_forget_only_fact(tmp_path):
    mock_client = MagicMock()

    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(
            message=MagicMock(content="")
        )
    ]

    memory = SummaryMemory(path=tmp_path, client=mock_client)

    memory.summary = "The user is vegetarian."
    memory.save_summary()

    memory.forget("The user is vegetarian.")

    assert memory.dump() == ""
    assert (tmp_path / "summary.txt").read_text(
        encoding="utf-8"
    ) == ""
