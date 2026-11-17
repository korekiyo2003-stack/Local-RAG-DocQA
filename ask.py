"""Run without installing: python ask.py <docs folder> ["question"]. See src/cli.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
