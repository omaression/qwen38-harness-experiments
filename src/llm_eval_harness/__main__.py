"""`python -m llm_eval_harness ...` is the same as the `llm-eval ...` console script."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
