import argparse
import uuid

from dotenv import load_dotenv

from .config import AppConfig
from .graph import run_agent


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the Deep Agents commerce graph.")
    parser.add_argument("task", nargs="+", help="User request")
    parser.add_argument("--session-id", default=None, help="Conversation session id")
    args = parser.parse_args()

    task = " ".join(args.task)
    session_id = args.session_id or str(uuid.uuid4())
    config = AppConfig.from_env()

    state = run_agent(task, session_id=session_id, config=config)
    print("Session:", session_id)
    print("Todos:", ", ".join(state.get("todos", [])) or "(none)")
    print("Result:\n", state.get("result", ""))


if __name__ == "__main__":
    main()
