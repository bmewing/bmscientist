"""Allow running bmscientist as `python -m bmscientist`."""
from bmscientist.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
