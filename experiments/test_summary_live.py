
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv

from dsa4213.memory.base import Session, Turn
from dsa4213.memory.summary import SummaryMemory

import os
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)


with TemporaryDirectory() as temp_dir:
    memory = SummaryMemory(
        path=Path(temp_dir),
        model="openai/gpt-4o-mini",
        client=client
    )

    # Step 1: Write the first fact
    session_1 = Session(
        session_id="session_1",
        date="2026-10-20",
        turns=(
            Turn(role="user", text="I am vegetarian."),
        )
    )

    memory.write(session_1)
    print("\n=== After Write 1 ===")
    print(memory.dump())

    # Step 2: Write additional facts
    session_2 = Session(
        session_id="session_2",
        date="2026-10-21",
        turns=(
            Turn(
                role="user",
                text="I like Japanese food and I study at NUS."
            ),
        )
    )

    memory.write(session_2)
    print("\n=== After Write 2 ===")
    print(memory.dump())

    # Step 3: Forget the targeted fact
    memory.forget("The user is vegetarian.")

    print("\n=== After Forget ===")
    print(memory.dump())

    # Step 4: Retrieve memory
    print("\n=== Retrieved Memory ===")
    print(memory.retrieve("What do you know about me?"))
